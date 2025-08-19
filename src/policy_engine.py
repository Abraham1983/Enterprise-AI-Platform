# Policy Engine - Rules, Thresholds, and Routing Decisions
# Supports JSON/YAML configuration with validation and approval workflows

import os
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

# Database
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class PolicyAction(Enum):
    AUTO_SEND = "auto_send"
    REVIEW_QUEUE = "review_queue"
    REJECT = "reject"
    ESCALATE = "escalate"
    HOLD = "hold"

class PolicySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RuleType(Enum):
    THRESHOLD = "threshold"
    VALIDATION = "validation"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    BUSINESS = "business"

# Database Models
class Policy(Base):
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    
    # Policy configuration
    rules = Column(JSON)  # List of rules
    actions = Column(JSON)  # Actions to take
    conditions = Column(JSON)  # Conditions for activation
    
    # Metadata
    enabled = Column(Boolean, default=True)
    priority = Column(Integer, default=100)  # Lower = higher priority
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)
    
    # Statistics
    triggered_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)

@dataclass
class PolicyRule:
    """Individual policy rule"""
    name: str
    type: RuleType
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex, exists
    value: Any
    severity: PolicySeverity = PolicySeverity.MEDIUM
    message: str = ""
    enabled: bool = True

@dataclass
class PolicyDecision:
    """Result of policy evaluation"""
    action: PolicyAction
    confidence: float
    triggered_rules: List[str]
    severity: PolicySeverity
    message: str
    metadata: Dict[str, Any]
    policy_name: str

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float
    suggested_fixes: List[str]

class PolicyEngine:
    """Main policy engine for rule evaluation and routing decisions"""
    
    def __init__(self, db_session: Session, config_path: str = "policies.yaml"):
        self.db = db_session
        self.config_path = config_path
        self.policies = {}
        self.default_policies = self._get_default_policies()
        self._load_policies()
    
    def _get_default_policies(self) -> Dict[str, Any]:
        """Get default policies for common scenarios"""
        
        return {
            "high_confidence_auto_send": {
                "name": "High Confidence Auto Send",
                "description": "Automatically send invoices with high confidence scores",
                "rules": [
                    {
                        "name": "confidence_threshold",
                        "type": "threshold",
                        "field": "confidence_score",
                        "operator": "gte",
                        "value": 0.9,
                        "severity": "low",
                        "message": "High confidence invoice"
                    },
                    {
                        "name": "amount_reasonable",
                        "type": "threshold",
                        "field": "total_amount",
                        "operator": "lte",
                        "value": 10000,
                        "severity": "medium",
                        "message": "Amount within reasonable limits"
                    }
                ],
                "actions": [
                    {
                        "type": "auto_send",
                        "conditions": ["all_rules_pass"]
                    }
                ],
                "enabled": True,
                "priority": 10
            },
            
            "amount_validation": {
                "name": "Amount Validation",
                "description": "Validate invoice amounts and flag anomalies",
                "rules": [
                    {
                        "name": "amount_exists",
                        "type": "validation",
                        "field": "total_amount",
                        "operator": "exists",
                        "value": True,
                        "severity": "high",
                        "message": "Total amount is required"
                    },
                    {
                        "name": "amount_positive",
                        "type": "validation",
                        "field": "total_amount",
                        "operator": "gt",
                        "value": 0,
                        "severity": "high",
                        "message": "Amount must be positive"
                    },
                    {
                        "name": "amount_not_excessive",
                        "type": "threshold",
                        "field": "total_amount",
                        "operator": "lte",
                        "value": 50000,
                        "severity": "medium",
                        "message": "Amount exceeds normal range"
                    }
                ],
                "actions": [
                    {
                        "type": "review_queue",
                        "conditions": ["any_rule_fails"]
                    }
                ],
                "enabled": True,
                "priority": 20
            },
            
            "client_validation": {
                "name": "Client Validation",
                "description": "Validate client information and requirements",
                "rules": [
                    {
                        "name": "client_email_exists",
                        "type": "validation",
                        "field": "client.email",
                        "operator": "exists",
                        "value": True,
                        "severity": "high",
                        "message": "Client email is required"
                    },
                    {
                        "name": "client_email_valid",
                        "type": "pattern",
                        "field": "client.email",
                        "operator": "regex",
                        "value": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                        "severity": "high",
                        "message": "Invalid email format"
                    },
                    {
                        "name": "client_name_exists",
                        "type": "validation",
                        "field": "client.name",
                        "operator": "exists",
                        "value": True,
                        "severity": "medium",
                        "message": "Client name is recommended"
                    }
                ],
                "actions": [
                    {
                        "type": "review_queue",
                        "conditions": ["high_severity_fails"]
                    }
                ],
                "enabled": True,
                "priority": 30
            },
            
            "duplicate_detection": {
                "name": "Duplicate Detection",
                "description": "Detect potential duplicate invoices",
                "rules": [
                    {
                        "name": "duplicate_check",
                        "type": "business",
                        "field": "invoice_number",
                        "operator": "custom",
                        "value": "check_duplicates",
                        "severity": "high",
                        "message": "Potential duplicate invoice detected"
                    }
                ],
                "actions": [
                    {
                        "type": "hold",
                        "conditions": ["any_rule_fails"]
                    }
                ],
                "enabled": True,
                "priority": 5
            }
        }
    
    def _load_policies(self):
        """Load policies from database and config file"""
        
        try:
            # Load from database
            db_policies = self.db.query(Policy).filter(Policy.enabled == True).all()
            for policy in db_policies:
                self.policies[policy.name] = {
                    'id': policy.id,
                    'name': policy.name,
                    'description': policy.description,
                    'rules': policy.rules,
                    'actions': policy.actions,
                    'conditions': policy.conditions,
                    'priority': policy.priority,
                    'enabled': policy.enabled
                }
            
            # Add default policies if not present
            for name, policy in self.default_policies.items():
                if name not in self.policies:
                    self.policies[name] = policy
            
            logger.info(f"Loaded {len(self.policies)} policies")
            
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            # Fall back to default policies
            self.policies = self.default_policies.copy()
    
    def evaluate_rule(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a single rule against data"""
        
        try:
            field = rule['field']
            operator = rule['operator']
            expected_value = rule['value']
            
            # Get field value from data (supports nested fields like 'client.email')
            field_value = self._get_nested_value(data, field)
            
            # Evaluate based on operator
            if operator == "exists":
                result = field_value is not None and field_value != ""
            elif operator == "eq":
                result = field_value == expected_value
            elif operator == "ne":
                result = field_value != expected_value
            elif operator == "gt":
                result = field_value is not None and float(field_value) > float(expected_value)
            elif operator == "lt":
                result = field_value is not None and float(field_value) < float(expected_value)
            elif operator == "gte":
                result = field_value is not None and float(field_value) >= float(expected_value)
            elif operator == "lte":
                result = field_value is not None and float(field_value) <= float(expected_value)
            elif operator == "in":
                result = field_value in expected_value
            elif operator == "not_in":
                result = field_value not in expected_value
            elif operator == "contains":
                result = field_value is not None and str(expected_value) in str(field_value)
            elif operator == "regex":
                result = field_value is not None and bool(re.match(expected_value, str(field_value)))
            elif operator == "custom":
                # Custom business logic
                result = self._evaluate_custom_rule(expected_value, field_value, data)
            else:
                logger.warning(f"Unknown operator: {operator}")
                result = False
            
            message = rule.get('message', f"Rule {rule['name']} evaluation")
            return result, message
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.get('name', 'unknown')}: {e}")
            return False, f"Rule evaluation error: {str(e)}"
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        
        try:
            keys = field.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _evaluate_custom_rule(self, rule_name: str, field_value: Any, data: Dict[str, Any]) -> bool:
        """Evaluate custom business rules"""
        
        try:
            if rule_name == "validate_tax_calculation":
                # Validate tax calculation
                subtotal = data.get('subtotal', 0)
                tax_rate = data.get('tax_rate', 0)
                tax_amount = data.get('tax_amount', 0)
                
                if subtotal and tax_rate:
                    expected_tax = subtotal * tax_rate
                    tolerance = 0.01  # 1 cent tolerance
                    return abs(tax_amount - expected_tax) <= tolerance
                
                return True  # Skip validation if data missing
            
            elif rule_name == "check_duplicates":
                # Check for duplicate invoices (simplified)
                invoice_number = data.get('invoice_number')
                client_name = data.get('client', {}).get('name')
                
                if invoice_number and client_name:
                    # This would query your database for duplicates
                    return True  # No duplicates found
                
                return True
            
            else:
                logger.warning(f"Unknown custom rule: {rule_name}")
                return True
                
        except Exception as e:
            logger.error(f"Custom rule evaluation failed: {e}")
            return True  # Default to pass on error
    
    def evaluate_policy(self, policy_name: str, data: Dict[str, Any]) -> PolicyDecision:
        """Evaluate a specific policy against data"""
        
        if policy_name not in self.policies:
            return PolicyDecision(
                action=PolicyAction.REVIEW_QUEUE,
                confidence=0.0,
                triggered_rules=[],
                severity=PolicySeverity.MEDIUM,
                message=f"Policy not found: {policy_name}",
                metadata={},
                policy_name=policy_name
            )
        
        policy = self.policies[policy_name]
        
        if not policy.get('enabled', True):
            return PolicyDecision(
                action=PolicyAction.AUTO_SEND,
                confidence=1.0,
                triggered_rules=[],
                severity=PolicySeverity.LOW,
                message="Policy disabled",
                metadata={},
                policy_name=policy_name
            )
        
        # Evaluate all rules
        rule_results = []
        triggered_rules = []
        max_severity = PolicySeverity.LOW
        
        for rule in policy.get('rules', []):
            if not rule.get('enabled', True):
                continue
            
            passed, message = self.evaluate_rule(rule, data)
            rule_results.append({
                'name': rule['name'],
                'passed': passed,
                'message': message,
                'severity': rule.get('severity', 'medium')
            })
            
            if not passed:
                triggered_rules.append(rule['name'])
                rule_severity = PolicySeverity(rule.get('severity', 'medium'))
                if rule_severity.value == 'critical' or (rule_severity.value == 'high' and max_severity.value != 'critical'):
                    max_severity = rule_severity
                elif rule_severity.value == 'medium' and max_severity.value in ['low']:
                    max_severity = rule_severity
        
        # Determine action based on rules and conditions
        action = self._determine_action(policy, rule_results, triggered_rules)
        
        # Calculate confidence
        total_rules = len([r for r in policy.get('rules', []) if r.get('enabled', True)])
        passed_rules = len([r for r in rule_results if r['passed']])
        confidence = passed_rules / total_rules if total_rules > 0 else 1.0
        
        # Update policy statistics
        self._update_policy_stats(policy_name, len(triggered_rules) > 0)
        
        return PolicyDecision(
            action=action,
            confidence=confidence,
            triggered_rules=triggered_rules,
            severity=max_severity,
            message=f"Policy evaluation: {len(triggered_rules)} rules triggered",
            metadata={
                'rule_results': rule_results,
                'total_rules': total_rules,
                'passed_rules': passed_rules
            },
            policy_name=policy_name
        )
    
    def _determine_action(self, policy: Dict[str, Any], rule_results: List[Dict], triggered_rules: List[str]) -> PolicyAction:
        """Determine action based on policy rules and conditions"""
        
        actions = policy.get('actions', [])
        
        for action_config in actions:
            action_type = action_config.get('type', 'review_queue')
            conditions = action_config.get('conditions', [])
            
            # Check conditions
            condition_met = False
            
            for condition in conditions:
                if condition == "all_rules_pass":
                    condition_met = len(triggered_rules) == 0
                elif condition == "any_rule_fails":
                    condition_met = len(triggered_rules) > 0
                elif condition == "high_severity_fails":
                    high_severity_fails = any(
                        not r['passed'] and r['severity'] in ['high', 'critical']
                        for r in rule_results
                    )
                    condition_met = high_severity_fails
                
                if condition_met:
                    break
            
            if condition_met:
                try:
                    return PolicyAction(action_type)
                except ValueError:
                    logger.warning(f"Unknown action type: {action_type}")
                    return PolicyAction.REVIEW_QUEUE
        
        # Default action
        return PolicyAction.REVIEW_QUEUE
    
    def _update_policy_stats(self, policy_name: str, triggered: bool):
        """Update policy statistics"""
        
        try:
            policy = self.db.query(Policy).filter(Policy.name == policy_name).first()
            if policy:
                if triggered:
                    policy.triggered_count += 1
                    policy.last_triggered = datetime.utcnow()
                self.db.commit()
        except Exception as e:
            logger.error(f"Failed to update policy stats: {e}")
    
    def evaluate_all_policies(self, data: Dict[str, Any]) -> List[PolicyDecision]:
        """Evaluate all enabled policies against data"""
        
        decisions = []
        
        # Sort policies by priority
        sorted_policies = sorted(
            self.policies.items(),
            key=lambda x: x[1].get('priority', 100)
        )
        
        for policy_name, policy in sorted_policies:
            if policy.get('enabled', True):
                decision = self.evaluate_policy(policy_name, data)
                decisions.append(decision)
                
                # Stop on critical actions
                if decision.action in [PolicyAction.REJECT, PolicyAction.HOLD]:
                    break
        
        return decisions
    
    def get_routing_decision(self, data: Dict[str, Any]) -> PolicyDecision:
        """Get final routing decision based on all policies"""
        
        decisions = self.evaluate_all_policies(data)
        
        if not decisions:
            return PolicyDecision(
                action=PolicyAction.REVIEW_QUEUE,
                confidence=0.5,
                triggered_rules=[],
                severity=PolicySeverity.MEDIUM,
                message="No policies evaluated",
                metadata={},
                policy_name="default"
            )
        
        # Determine final action (most restrictive wins)
        action_priority = {
            PolicyAction.REJECT: 0,
            PolicyAction.HOLD: 1,
            PolicyAction.ESCALATE: 2,
            PolicyAction.REVIEW_QUEUE: 3,
            PolicyAction.AUTO_SEND: 4
        }
        
        final_decision = min(decisions, key=lambda d: action_priority.get(d.action, 3))
        
        # Aggregate metadata
        all_triggered_rules = []
        all_policies = []
        min_confidence = 1.0
        max_severity = PolicySeverity.LOW
        
        for decision in decisions:
            all_triggered_rules.extend(decision.triggered_rules)
            all_policies.append(decision.policy_name)
            min_confidence = min(min_confidence, decision.confidence)
            if decision.severity.value == 'critical' or (decision.severity.value == 'high' and max_severity.value != 'critical'):
                max_severity = decision.severity
            elif decision.severity.value == 'medium' and max_severity.value == 'low':
                max_severity = decision.severity
        
        return PolicyDecision(
            action=final_decision.action,
            confidence=min_confidence,
            triggered_rules=list(set(all_triggered_rules)),
            severity=max_severity,
            message=f"Evaluated {len(decisions)} policies, {len(all_triggered_rules)} rules triggered",
            metadata={
                'evaluated_policies': all_policies,
                'individual_decisions': [asdict(d) for d in decisions]
            },
            policy_name="aggregated"
        )
    
    def validate_invoice_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate invoice data against policies"""
        
        decision = self.get_routing_decision(data)
        
        errors = []
        warnings = []
        suggested_fixes = []
        
        # Extract errors and warnings from triggered rules
        for policy_decision in decision.metadata.get('individual_decisions', []):
            for rule_result in policy_decision.get('metadata', {}).get('rule_results', []):
                if not rule_result['passed']:
                    if rule_result['severity'] in ['high', 'critical']:
                        errors.append(rule_result['message'])
                        suggested_fixes.append(f"Fix: {rule_result['name']}")
                    else:
                        warnings.append(rule_result['message'])
        
        is_valid = decision.action not in [PolicyAction.REJECT, PolicyAction.HOLD]
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            confidence=decision.confidence,
            suggested_fixes=suggested_fixes
        )
    
    def get_all_policies(self) -> Dict[str, Any]:
        """Get all policies"""
        return self.policies.copy()

# Example usage
if __name__ == "__main__":
    # This would be used with your actual database session
    print("Policy Engine initialized with rule-based validation and routing!")