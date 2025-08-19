# Review Queue - Human-in-the-loop workflow for invoice approval
# Handles low-confidence invoices, policy violations, and manual review

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import difflib

# Database
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"
    ESCALATED = "escalated"

class ReviewPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ReviewReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    POLICY_VIOLATION = "policy_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    MANUAL_REVIEW = "manual_review"
    AMOUNT_THRESHOLD = "amount_threshold"
    CLIENT_RISK = "client_risk"
    DUPLICATE_SUSPECTED = "duplicate_suspected"

# Database Models
class ReviewItem(Base):
    __tablename__ = "review_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Invoice information
    invoice_number = Column(String, index=True)
    invoice_data = Column(JSON)  # Original invoice data
    suggested_changes = Column(JSON)  # AI-suggested fixes
    
    # Review metadata
    reason = Column(SQLEnum(ReviewReason))
    priority = Column(SQLEnum(ReviewPriority), default=ReviewPriority.MEDIUM)
    status = Column(SQLEnum(ReviewStatus), default=ReviewStatus.PENDING)
    
    # Confidence and scoring
    confidence_score = Column(Float)
    risk_score = Column(Float)
    
    # Policy information
    triggered_policies = Column(JSON)  # List of triggered policy names
    policy_violations = Column(JSON)  # Detailed policy violations
    
    # Review workflow
    assigned_to = Column(String)  # User ID or email
    reviewed_by = Column(String)
    review_notes = Column(Text)
    review_decision = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime)
    reviewed_at = Column(DateTime)
    due_date = Column(DateTime)
    
    # Escalation
    escalated = Column(Boolean, default=False)
    escalated_at = Column(DateTime)
    escalated_to = Column(String)
    
    # Metadata
    metadata = Column(JSON)

@dataclass
class ReviewRequest:
    """Request to add item to review queue"""
    invoice_number: str
    invoice_data: Dict[str, Any]
    reason: ReviewReason
    priority: ReviewPriority = ReviewPriority.MEDIUM
    confidence_score: float = 0.0
    risk_score: float = 0.0
    triggered_policies: List[str] = None
    policy_violations: List[Dict[str, Any]] = None
    suggested_changes: Dict[str, Any] = None
    assigned_to: str = None
    due_hours: int = 24
    metadata: Dict[str, Any] = None

@dataclass
class ReviewDecision:
    """Review decision from human reviewer"""
    action: ReviewStatus
    notes: str = ""
    changes: Dict[str, Any] = None
    escalate: bool = False
    escalate_to: str = None

@dataclass
class ReviewSummary:
    """Summary of review queue status"""
    total_pending: int
    high_priority: int
    overdue: int
    avg_review_time_hours: float
    approval_rate: float
    by_reason: Dict[str, int]
    by_priority: Dict[str, int]

class ReviewQueue:
    """Main review queue management system"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def add_to_queue(self, request: ReviewRequest) -> int:
        """Add item to review queue"""
        
        try:
            # Check if item already exists
            existing = self.db.query(ReviewItem).filter(
                ReviewItem.invoice_number == request.invoice_number,
                ReviewItem.status == ReviewStatus.PENDING
            ).first()
            
            if existing:
                logger.warning(f"Invoice {request.invoice_number} already in review queue")
                return existing.id
            
            # Calculate due date
            due_date = datetime.utcnow() + timedelta(hours=request.due_hours)
            
            # Create review item
            review_item = ReviewItem(
                invoice_number=request.invoice_number,
                invoice_data=request.invoice_data,
                suggested_changes=request.suggested_changes or {},
                reason=request.reason,
                priority=request.priority,
                confidence_score=request.confidence_score,
                risk_score=request.risk_score,
                triggered_policies=request.triggered_policies or [],
                policy_violations=request.policy_violations or [],
                assigned_to=request.assigned_to,
                due_date=due_date,
                metadata=request.metadata or {}
            )
            
            self.db.add(review_item)
            self.db.commit()
            
            logger.info(f"Added invoice {request.invoice_number} to review queue (ID: {review_item.id})")
            return review_item.id
            
        except Exception as e:
            logger.error(f"Failed to add item to review queue: {e}")
            self.db.rollback()
            raise
    
    def get_queue_items(self, 
                       status: ReviewStatus = None,
                       assigned_to: str = None,
                       priority: ReviewPriority = None,
                       limit: int = 50,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """Get items from review queue with filtering"""
        
        try:
            query = self.db.query(ReviewItem)
            
            # Apply filters
            if status:
                query = query.filter(ReviewItem.status == status)
            if assigned_to:
                query = query.filter(ReviewItem.assigned_to == assigned_to)
            if priority:
                query = query.filter(ReviewItem.priority == priority)
            
            # Order by priority and creation time
            items = query.order_by(
                ReviewItem.priority,
                ReviewItem.created_at
            ).offset(offset).limit(limit).all()
            
            # Convert to dict format
            result = []
            for item in items:
                result.append({
                    'id': item.id,
                    'invoice_number': item.invoice_number,
                    'invoice_data': item.invoice_data,
                    'suggested_changes': item.suggested_changes,
                    'reason': item.reason.value,
                    'priority': item.priority.value,
                    'status': item.status.value,
                    'confidence_score': item.confidence_score,
                    'risk_score': item.risk_score,
                    'triggered_policies': item.triggered_policies,
                    'policy_violations': item.policy_violations,
                    'assigned_to': item.assigned_to,
                    'reviewed_by': item.reviewed_by,
                    'review_notes': item.review_notes,
                    'created_at': item.created_at.isoformat(),
                    'assigned_at': item.assigned_at.isoformat() if item.assigned_at else None,
                    'reviewed_at': item.reviewed_at.isoformat() if item.reviewed_at else None,
                    'due_date': item.due_date.isoformat() if item.due_date else None,
                    'is_overdue': item.due_date and datetime.utcnow() > item.due_date,
                    'escalated': item.escalated,
                    'metadata': item.metadata
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get queue items: {e}")
            return []
    
    def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get specific review item"""
        
        try:
            item = self.db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
            
            if not item:
                return None
            
            # Generate diff if suggested changes exist
            diff_html = None
            if item.suggested_changes:
                diff_html = self._generate_diff(item.invoice_data, item.suggested_changes)
            
            return {
                'id': item.id,
                'invoice_number': item.invoice_number,
                'invoice_data': item.invoice_data,
                'suggested_changes': item.suggested_changes,
                'diff_html': diff_html,
                'reason': item.reason.value,
                'priority': item.priority.value,
                'status': item.status.value,
                'confidence_score': item.confidence_score,
                'risk_score': item.risk_score,
                'triggered_policies': item.triggered_policies,
                'policy_violations': item.policy_violations,
                'assigned_to': item.assigned_to,
                'reviewed_by': item.reviewed_by,
                'review_notes': item.review_notes,
                'review_decision': item.review_decision,
                'created_at': item.created_at.isoformat(),
                'assigned_at': item.assigned_at.isoformat() if item.assigned_at else None,
                'reviewed_at': item.reviewed_at.isoformat() if item.reviewed_at else None,
                'due_date': item.due_date.isoformat() if item.due_date else None,
                'is_overdue': item.due_date and datetime.utcnow() > item.due_date,
                'escalated': item.escalated,
                'escalated_at': item.escalated_at.isoformat() if item.escalated_at else None,
                'escalated_to': item.escalated_to,
                'metadata': item.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get review item {item_id}: {e}")
            return None
    
    def assign_item(self, item_id: int, assigned_to: str) -> bool:
        """Assign review item to user"""
        
        try:
            item = self.db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
            
            if not item:
                return False
            
            item.assigned_to = assigned_to
            item.assigned_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(f"Assigned review item {item_id} to {assigned_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign review item: {e}")
            self.db.rollback()
            return False
    
    def review_item(self, item_id: int, decision: ReviewDecision, reviewed_by: str) -> bool:
        """Process review decision"""
        
        try:
            item = self.db.query(ReviewItem).filter(ReviewItem.id == item_id).first()
            
            if not item:
                return False
            
            # Update review item
            item.status = decision.action
            item.reviewed_by = reviewed_by
            item.review_notes = decision.notes
            item.review_decision = json.dumps(asdict(decision))
            item.reviewed_at = datetime.utcnow()
            
            # Handle escalation
            if decision.escalate:
                item.escalated = True
                item.escalated_at = datetime.utcnow()
                item.escalated_to = decision.escalate_to
                item.priority = ReviewPriority.URGENT
            
            # Apply changes if approved with modifications
            if decision.action == ReviewStatus.APPROVED and decision.changes:
                # Merge changes into invoice data
                updated_data = item.invoice_data.copy()
                updated_data.update(decision.changes)
                item.invoice_data = updated_data
            
            self.db.commit()
            
            logger.info(f"Reviewed item {item_id}: {decision.action.value} by {reviewed_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to review item: {e}")
            self.db.rollback()
            return False
    
    def bulk_assign(self, item_ids: List[int], assigned_to: str) -> int:
        """Bulk assign multiple items"""
        
        try:
            updated = self.db.query(ReviewItem).filter(
                ReviewItem.id.in_(item_ids),
                ReviewItem.status == ReviewStatus.PENDING
            ).update({
                'assigned_to': assigned_to,
                'assigned_at': datetime.utcnow()
            }, synchronize_session=False)
            
            self.db.commit()
            
            logger.info(f"Bulk assigned {updated} items to {assigned_to}")
            return updated
            
        except Exception as e:
            logger.error(f"Failed to bulk assign items: {e}")
            self.db.rollback()
            return 0
    
    def escalate_overdue(self, escalate_to: str, hours_overdue: int = 24) -> int:
        """Escalate overdue items"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_overdue)
            
            overdue_items = self.db.query(ReviewItem).filter(
                ReviewItem.status == ReviewStatus.PENDING,
                ReviewItem.due_date < cutoff_time,
                ReviewItem.escalated == False
            ).all()
            
            escalated_count = 0
            for item in overdue_items:
                item.escalated = True
                item.escalated_at = datetime.utcnow()
                item.escalated_to = escalate_to
                item.priority = ReviewPriority.URGENT
                escalated_count += 1
            
            self.db.commit()
            
            logger.info(f"Escalated {escalated_count} overdue items to {escalate_to}")
            return escalated_count
            
        except Exception as e:
            logger.error(f"Failed to escalate overdue items: {e}")
            self.db.rollback()
            return 0
    
    def get_summary(self) -> ReviewSummary:
        """Get review queue summary statistics"""
        
        try:
            # Total pending
            total_pending = self.db.query(ReviewItem).filter(
                ReviewItem.status == ReviewStatus.PENDING
            ).count()
            
            # High priority
            high_priority = self.db.query(ReviewItem).filter(
                ReviewItem.status == ReviewStatus.PENDING,
                ReviewItem.priority.in_([ReviewPriority.HIGH, ReviewPriority.URGENT])
            ).count()
            
            # Overdue
            overdue = self.db.query(ReviewItem).filter(
                ReviewItem.status == ReviewStatus.PENDING,
                ReviewItem.due_date < datetime.utcnow()
            ).count()
            
            # Average review time
            completed_items = self.db.query(ReviewItem).filter(
                ReviewItem.status.in_([ReviewStatus.APPROVED, ReviewStatus.REJECTED]),
                ReviewItem.reviewed_at.isnot(None)
            ).all()
            
            if completed_items:
                total_hours = sum(
                    (item.reviewed_at - item.created_at).total_seconds() / 3600
                    for item in completed_items
                )
                avg_review_time = total_hours / len(completed_items)
            else:
                avg_review_time = 0.0
            
            # Approval rate
            approved_count = self.db.query(ReviewItem).filter(
                ReviewItem.status == ReviewStatus.APPROVED
            ).count()
            
            total_reviewed = len(completed_items)
            approval_rate = approved_count / total_reviewed if total_reviewed > 0 else 0.0
            
            # By reason
            by_reason = {}
            for reason in ReviewReason:
                count = self.db.query(ReviewItem).filter(
                    ReviewItem.reason == reason,
                    ReviewItem.status == ReviewStatus.PENDING
                ).count()
                by_reason[reason.value] = count
            
            # By priority
            by_priority = {}
            for priority in ReviewPriority:
                count = self.db.query(ReviewItem).filter(
                    ReviewItem.priority == priority,
                    ReviewItem.status == ReviewStatus.PENDING
                ).count()
                by_priority[priority.value] = count
            
            return ReviewSummary(
                total_pending=total_pending,
                high_priority=high_priority,
                overdue=overdue,
                avg_review_time_hours=avg_review_time,
                approval_rate=approval_rate,
                by_reason=by_reason,
                by_priority=by_priority
            )
            
        except Exception as e:
            logger.error(f"Failed to get review summary: {e}")
            return ReviewSummary(
                total_pending=0,
                high_priority=0,
                overdue=0,
                avg_review_time_hours=0.0,
                approval_rate=0.0,
                by_reason={},
                by_priority={}
            )
    
    def _generate_diff(self, original: Dict[str, Any], suggested: Dict[str, Any]) -> str:
        """Generate HTML diff between original and suggested data"""
        
        try:
            # Convert to formatted JSON strings
            original_str = json.dumps(original, indent=2, sort_keys=True)
            suggested_str = json.dumps(suggested, indent=2, sort_keys=True)
            
            # Generate diff
            diff = difflib.unified_diff(
                original_str.splitlines(keepends=True),
                suggested_str.splitlines(keepends=True),
                fromfile='Original',
                tofile='Suggested',
                lineterm=''
            )
            
            # Convert to HTML
            html_diff = []
            for line in diff:
                if line.startswith('+++') or line.startswith('---'):
                    html_diff.append(f'<div class="diff-header">{line}</div>')
                elif line.startswith('@@'):
                    html_diff.append(f'<div class="diff-hunk">{line}</div>')
                elif line.startswith('+'):
                    html_diff.append(f'<div class="diff-added">{line}</div>')
                elif line.startswith('-'):
                    html_diff.append(f'<div class="diff-removed">{line}</div>')
                else:
                    html_diff.append(f'<div class="diff-context">{line}</div>')
            
            return '\n'.join(html_diff)
            
        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return "Diff generation failed"
    
    def cleanup_old_items(self, days_old: int = 30) -> int:
        """Clean up old completed review items"""
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            deleted = self.db.query(ReviewItem).filter(
                ReviewItem.status.in_([ReviewStatus.APPROVED, ReviewStatus.REJECTED]),
                ReviewItem.reviewed_at < cutoff_date
            ).delete(synchronize_session=False)
            
            self.db.commit()
            
            logger.info(f"Cleaned up {deleted} old review items")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old items: {e}")
            self.db.rollback()
            return 0

# Helper functions for integration
def create_review_request_from_policy_decision(
    invoice_data: Dict[str, Any],
    policy_decision,
    confidence_score: float = 0.0
) -> ReviewRequest:
    """Create review request from policy decision"""
    
    # Determine reason
    if policy_decision.action.value == "review_queue":
        if confidence_score < 0.5:
            reason = ReviewReason.LOW_CONFIDENCE
        elif policy_decision.triggered_rules:
            reason = ReviewReason.POLICY_VIOLATION
        else:
            reason = ReviewReason.MANUAL_REVIEW
    else:
        reason = ReviewReason.POLICY_VIOLATION
    
    # Determine priority
    if policy_decision.severity.value == "critical":
        priority = ReviewPriority.URGENT
    elif policy_decision.severity.value == "high":
        priority = ReviewPriority.HIGH
    elif policy_decision.severity.value == "medium":
        priority = ReviewPriority.MEDIUM
    else:
        priority = ReviewPriority.LOW
    
    # Calculate risk score
    risk_score = 1.0 - confidence_score
    if policy_decision.triggered_rules:
        risk_score += len(policy_decision.triggered_rules) * 0.1
    
    return ReviewRequest(
        invoice_number=invoice_data.get('invoice_number', 'UNKNOWN'),
        invoice_data=invoice_data,
        reason=reason,
        priority=priority,
        confidence_score=confidence_score,
        risk_score=min(risk_score, 1.0),
        triggered_policies=[policy_decision.policy_name],
        policy_violations=[{
            'policy': policy_decision.policy_name,
            'triggered_rules': policy_decision.triggered_rules,
            'severity': policy_decision.severity.value,
            'message': policy_decision.message
        }],
        metadata={
            'policy_decision': asdict(policy_decision)
        }
    )

# Example usage
if __name__ == "__main__":
    # This would be used with your actual database session
    print("Review Queue initialized with human-in-the-loop workflow!")