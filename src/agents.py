# AI Agents - Multi-model LLM Router with Ollama/Qwen/OpenAI Support
# Agentic workflows for reconciliation, dunning, pricing, anomaly detection, and analysis

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import httpx
import openai

# Try to import optional dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    OLLAMA = "ollama"
    QWEN_NATIVE = "qwen_native"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class TaskType(Enum):
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TOOL_CALLING = "tool_calling"

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    provider: ModelProvider
    model_name: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30
    max_tokens: int = 4000
    temperature: float = 0.3
    priority: int = 1  # Lower = higher priority
    cost_per_token: float = 0.0
    context_length: int = 8000

@dataclass
class LLMResponse:
    """Response from LLM model"""
    content: str
    provider: ModelProvider
    model: str
    tokens_used: int
    response_time: float
    cost: float
    success: bool
    error: Optional[str] = None

class LLMRouter:
    """Intelligent router for multiple LLM providers with fallback"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.fallback_chain = []
        self.request_counts = {}
        self.error_counts = {}
        
        # Initialize models
        self._setup_models()
        
        # Setup fallback chain
        self._setup_fallback_chain()
    
    def _setup_models(self):
        """Setup available models"""
        
        # Ollama models
        if OLLAMA_AVAILABLE and self.config.get('ollama', {}).get('enabled', False):
            ollama_config = self.config['ollama']
            for model_name in ollama_config.get('models', ['llama3:8b', 'qwen2.5:7b']):
                self.models[f"ollama_{model_name}"] = ModelConfig(
                    provider=ModelProvider.OLLAMA,
                    model_name=model_name,
                    endpoint=ollama_config.get('endpoint', 'http://localhost:11434'),
                    timeout=ollama_config.get('timeout', 30),
                    priority=1,
                    cost_per_token=0.0,
                    context_length=ollama_config.get('context_length', 8000)
                )
        
        # Qwen native models
        if TRANSFORMERS_AVAILABLE and self.config.get('qwen', {}).get('enabled', False):
            qwen_config = self.config['qwen']
            for model_name in qwen_config.get('models', ['Qwen2.5-7B-Instruct']):
                self.models[f"qwen_{model_name}"] = ModelConfig(
                    provider=ModelProvider.QWEN_NATIVE,
                    model_name=model_name,
                    timeout=qwen_config.get('timeout', 45),
                    priority=1,
                    cost_per_token=0.0,
                    context_length=qwen_config.get('context_length', 32000)
                )
        
        # OpenAI models
        if self.config.get('openai', {}).get('api_key'):
            openai_config = self.config['openai']
            models = openai_config.get('models', {
                'gpt-4o-mini': {'cost': 0.00015, 'context': 128000},
                'gpt-4': {'cost': 0.03, 'context': 8000}
            })
            
            for model_name, model_info in models.items():
                self.models[f"openai_{model_name}"] = ModelConfig(
                    provider=ModelProvider.OPENAI,
                    model_name=model_name,
                    api_key=openai_config['api_key'],
                    timeout=openai_config.get('timeout', 30),
                    priority=3,  # Lower priority (fallback)
                    cost_per_token=model_info.get('cost', 0.001),
                    context_length=model_info.get('context', 8000)
                )
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def _setup_fallback_chain(self):
        """Setup fallback chain based on priority and availability"""
        
        # Sort models by priority
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1].priority
        )
        
        self.fallback_chain = [model_id for model_id, _ in sorted_models]
        logger.info(f"Fallback chain: {self.fallback_chain}")
    
    async def generate(self, 
                      prompt: str,
                      task_type: TaskType = TaskType.GENERATION,
                      context_length: Optional[int] = None,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      preferred_model: Optional[str] = None) -> LLMResponse:
        """Generate response using best available model"""
        
        # Select model
        selected_model = self._select_model(
            task_type=task_type,
            context_length=context_length or len(prompt.split()),
            preferred_model=preferred_model
        )
        
        if not selected_model:
            return LLMResponse(
                content="No available models",
                provider=ModelProvider.OPENAI,
                model="none",
                tokens_used=0,
                response_time=0,
                cost=0,
                success=False,
                error="No models available"
            )
        
        # Try models in fallback order
        for model_id in ([selected_model] + self.fallback_chain):
            if model_id not in self.models:
                continue
                
            try:
                response = await self._call_model(
                    model_id,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response.success:
                    self._update_success_stats(model_id)
                    return response
                else:
                    self._update_error_stats(model_id, response.error)
                    
            except Exception as e:
                logger.error(f"Model {model_id} failed: {e}")
                self._update_error_stats(model_id, str(e))
                continue
        
        # All models failed
        return LLMResponse(
            content="All models failed",
            provider=ModelProvider.OPENAI,
            model="fallback",
            tokens_used=0,
            response_time=0,
            cost=0,
            success=False,
            error="All models failed"
        )
    
    def _select_model(self, 
                     task_type: TaskType,
                     context_length: int,
                     preferred_model: Optional[str] = None) -> Optional[str]:
        """Select best model for task"""
        
        if preferred_model and preferred_model in self.models:
            return preferred_model
        
        # Filter models by capability
        suitable_models = []
        
        for model_id, config in self.models.items():
            # Check context length
            if context_length > config.context_length:
                continue
            
            # Check availability (error rate)
            error_rate = self._get_error_rate(model_id)
            if error_rate > 0.5:  # Skip if >50% error rate
                continue
            
            suitable_models.append((model_id, config))
        
        if not suitable_models:
            return None
        
        # Sort by priority and select best
        suitable_models.sort(key=lambda x: x[1].priority)
        return suitable_models[0][0]
    
    async def _call_model(self, 
                         model_id: str,
                         prompt: str,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> LLMResponse:
        """Call specific model"""
        
        config = self.models[model_id]
        start_time = time.time()
        
        try:
            if config.provider == ModelProvider.OLLAMA:
                return await self._call_ollama(config, prompt, max_tokens, temperature)
            elif config.provider == ModelProvider.QWEN_NATIVE:
                return await self._call_qwen_native(config, prompt, max_tokens, temperature)
            elif config.provider == ModelProvider.OPENAI:
                return await self._call_openai(config, prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
                
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content="",
                provider=config.provider,
                model=config.model_name,
                tokens_used=0,
                response_time=response_time,
                cost=0,
                success=False,
                error=str(e)
            )
    
    async def _call_ollama(self, config: ModelConfig, prompt: str, max_tokens: Optional[int], temperature: Optional[float]) -> LLMResponse:
        """Call Ollama API"""
        
        start_time = time.time()
        
        try:
            client = ollama.AsyncClient(host=config.endpoint)
            
            response = await client.chat(
                model=config.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_predict': max_tokens or config.max_tokens,
                    'temperature': temperature or config.temperature,
                    'top_p': 0.9,
                }
            )
            
            response_time = time.time() - start_time
            content = response['message']['content']
            tokens_used = response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
            
            return LLMResponse(
                content=content,
                provider=config.provider,
                model=config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=0,  # Local models are free
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            raise Exception(f"Ollama call failed: {e}")
    
    async def _call_qwen_native(self, config: ModelConfig, prompt: str, max_tokens: Optional[int], temperature: Optional[float]) -> LLMResponse:
        """Call Qwen native model"""
        
        start_time = time.time()
        
        try:
            # Load model and tokenizer (cache them in production)
            tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{config.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                f"Qwen/{config.model_name}",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens or config.max_tokens,
                    temperature=temperature or config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            content = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            response_time = time.time() - start_time
            tokens_used = len(outputs[0])
            
            return LLMResponse(
                content=content,
                provider=config.provider,
                model=config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=0,  # Local models are free
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            raise Exception(f"Qwen native call failed: {e}")
    
    async def _call_openai(self, config: ModelConfig, prompt: str, max_tokens: Optional[int], temperature: Optional[float]) -> LLMResponse:
        """Call OpenAI API"""
        
        start_time = time.time()
        
        try:
            openai.api_key = config.api_key
            
            response = await openai.ChatCompletion.acreate(
                model=config.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens or config.max_tokens,
                temperature=temperature or config.temperature,
                timeout=config.timeout
            )
            
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = tokens_used * config.cost_per_token
            
            return LLMResponse(
                content=content,
                provider=config.provider,
                model=config.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                cost=cost,
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            raise Exception(f"OpenAI call failed: {e}")
    
    def _update_success_stats(self, model_id: str):
        """Update success statistics"""
        if model_id not in self.request_counts:
            self.request_counts[model_id] = 0
        self.request_counts[model_id] += 1
    
    def _update_error_stats(self, model_id: str, error: str):
        """Update error statistics"""
        if model_id not in self.error_counts:
            self.error_counts[model_id] = 0
        self.error_counts[model_id] += 1
        logger.warning(f"Model {model_id} error: {error}")
    
    def _get_error_rate(self, model_id: str) -> float:
        """Get error rate for model"""
        requests = self.request_counts.get(model_id, 0)
        errors = self.error_counts.get(model_id, 0)
        
        if requests == 0:
            return 0.0
        
        return errors / (requests + errors)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        
        stats = {
            'models': {},
            'total_requests': sum(self.request_counts.values()),
            'total_errors': sum(self.error_counts.values())
        }
        
        for model_id in self.models:
            stats['models'][model_id] = {
                'requests': self.request_counts.get(model_id, 0),
                'errors': self.error_counts.get(model_id, 0),
                'error_rate': self._get_error_rate(model_id),
                'available': self._get_error_rate(model_id) < 0.5
            }
        
        return stats

# AI Agents using the LLM Router
class BaseAgent:
    """Base class for AI agents"""
    
    def __init__(self, llm_router: LLMRouter, name: str):
        self.llm_router = llm_router
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run agent with goal and context"""
        raise NotImplementedError
    
    async def _generate_response(self, prompt: str, task_type: TaskType = TaskType.GENERATION) -> str:
        """Generate response using LLM router"""
        
        response = await self.llm_router.generate(prompt, task_type)
        
        if response.success:
            return response.content
        else:
            self.logger.error(f"LLM generation failed: {response.error}")
            return f"Error: {response.error}"

class ReconciliationAgent(BaseAgent):
    """Agent for reconciling POs/time logs vs invoices"""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(llm_router, "reconciliation")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reconcile purchase orders and time logs against invoices"""
        
        invoices = context.get('invoices', [])
        pos = context.get('purchase_orders', [])
        time_logs = context.get('time_logs', [])
        
        prompt = f"""
        Analyze and reconcile the following data:
        
        Goal: {goal}
        
        Invoices: {json.dumps(invoices, indent=2)}
        Purchase Orders: {json.dumps(pos, indent=2)}
        Time Logs: {json.dumps(time_logs, indent=2)}
        
        Please identify:
        1. Discrepancies between invoices and POs
        2. Time log entries that don't match invoices
        3. Missing invoices for completed work
        4. Billing rate inconsistencies
        5. Recommendations for resolution
        
        Format your response as JSON with sections for discrepancies, missing_items, recommendations.
        """
        
        response = await self._generate_response(prompt, TaskType.ANALYSIS)
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"error": "Failed to parse response", "raw_response": response}
        
        return {
            "agent": self.name,
            "goal": goal,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

class DunningAgent(BaseAgent):
    """Agent for automated follow-ups with grounding and payment links"""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(llm_router, "dunning")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate dunning emails with context and payment links"""
        
        invoice = context.get('invoice', {})
        client = context.get('client', {})
        overdue_days = context.get('overdue_days', 0)
        previous_emails = context.get('previous_emails', [])
        
        prompt = f"""
        Generate a professional dunning email for an overdue invoice.
        
        Goal: {goal}
        
        Invoice Details:
        - Invoice Number: {invoice.get('invoice_number')}
        - Amount: ${invoice.get('total_amount', 0):.2f}
        - Due Date: {invoice.get('due_date')}
        - Days Overdue: {overdue_days}
        
        Client Details:
        - Name: {client.get('name')}
        - Email: {client.get('email')}
        - Company: {client.get('company')}
        
        Previous Communications: {len(previous_emails)} emails sent
        
        Requirements:
        - Professional but firm tone
        - Reference specific invoice details
        - Include payment options
        - Escalation notice if severely overdue
        - Clear call to action
        
        Generate email subject and body.
        """
        
        response = await self._generate_response(prompt, TaskType.GENERATION)
        
        # Add payment links (mock implementation)
        payment_link = f"https://payments.example.com/invoice/{invoice.get('invoice_number')}"
        crypto_link = f"https://crypto.example.com/pay/{invoice.get('invoice_number')}"
        
        return {
            "agent": self.name,
            "goal": goal,
            "result": {
                "email_content": response,
                "payment_link": payment_link,
                "crypto_payment_link": crypto_link,
                "overdue_days": overdue_days,
                "escalation_level": "high" if overdue_days > 60 else "medium" if overdue_days > 30 else "low"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

class PricingAdvisor(BaseAgent):
    """Agent for rate suggestions and pricing optimization"""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(llm_router, "pricing_advisor")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze pricing and suggest rate optimizations"""
        
        historical_rates = context.get('historical_rates', [])
        market_data = context.get('market_data', {})
        client_profile = context.get('client_profile', {})
        project_type = context.get('project_type', 'general')
        
        prompt = f"""
        Analyze pricing strategy and provide rate recommendations.
        
        Goal: {goal}
        
        Historical Rates: {json.dumps(historical_rates, indent=2)}
        Market Data: {json.dumps(market_data, indent=2)}
        Client Profile: {json.dumps(client_profile, indent=2)}
        Project Type: {project_type}
        
        Please analyze:
        1. Current rate competitiveness
        2. Market trends and benchmarks
        3. Client-specific pricing factors
        4. Recommended rate adjustments
        5. Pricing strategy recommendations
        
        Format response as JSON with current_analysis, market_comparison, recommendations.
        """
        
        response = await self._generate_response(prompt, TaskType.ANALYSIS)
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"error": "Failed to parse response", "raw_response": response}
        
        return {
            "agent": self.name,
            "goal": goal,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

class AnomalyAgent(BaseAgent):
    """Agent for triaging anomalies and suggesting actions"""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(llm_router, "anomaly")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze anomalies and suggest triage actions"""
        
        anomalies = context.get('anomalies', [])
        historical_context = context.get('historical_context', {})
        
        prompt = f"""
        Analyze the following anomalies and provide triage recommendations.
        
        Goal: {goal}
        
        Detected Anomalies: {json.dumps(anomalies, indent=2)}
        Historical Context: {json.dumps(historical_context, indent=2)}
        
        For each anomaly, provide:
        1. Severity assessment (critical, high, medium, low)
        2. Likely root cause
        3. Recommended immediate actions
        4. Whether human review is needed
        5. Priority order for investigation
        
        Format response as JSON with anomaly_analysis array and overall_recommendations.
        """
        
        response = await self._generate_response(prompt, TaskType.ANALYSIS)
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"error": "Failed to parse response", "raw_response": response}
        
        return {
            "agent": self.name,
            "goal": goal,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

class AnalystAgent(BaseAgent):
    """Agent for daily analysis and change briefs"""
    
    def __init__(self, llm_router: LLMRouter):
        super().__init__(llm_router, "analyst")
    
    async def run(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate daily analysis brief of what changed"""
        
        daily_metrics = context.get('daily_metrics', {})
        previous_metrics = context.get('previous_metrics', {})
        recent_activities = context.get('recent_activities', [])
        alerts = context.get('alerts', [])
        
        prompt = f"""
        Generate a daily business analysis brief.
        
        Goal: {goal}
        
        Today's Metrics: {json.dumps(daily_metrics, indent=2)}
        Previous Day Metrics: {json.dumps(previous_metrics, indent=2)}
        Recent Activities: {json.dumps(recent_activities, indent=2)}
        Active Alerts: {json.dumps(alerts, indent=2)}
        
        Provide:
        1. Key changes from yesterday
        2. Notable trends and patterns
        3. Areas requiring attention
        4. Performance highlights
        5. Actionable recommendations
        
        Format as an executive briefing suitable for management review.
        """
        
        response = await self._generate_response(prompt, TaskType.ANALYSIS)
        
        return {
            "agent": self.name,
            "goal": goal,
            "result": {
                "brief": response,
                "generated_date": datetime.utcnow().date().isoformat(),
                "metrics_analyzed": len(daily_metrics),
                "activities_reviewed": len(recent_activities),
                "alerts_count": len(alerts)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

# Agent factory
class AgentFactory:
    """Factory for creating and managing agents"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_router = LLMRouter(llm_config)
        self.agents = {
            'reconciliation': ReconciliationAgent(self.llm_router),
            'dunning': DunningAgent(self.llm_router),
            'pricing_advisor': PricingAdvisor(self.llm_router),
            'anomaly': AnomalyAgent(self.llm_router),
            'analyst': AnalystAgent(self.llm_router)
        }
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List available agents"""
        return list(self.agents.keys())
    
    async def run_agent(self, agent_name: str, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run specific agent"""
        
        agent = self.get_agent(agent_name)
        if not agent:
            return {
                "error": f"Agent '{agent_name}' not found",
                "available_agents": self.list_agents()
            }
        
        try:
            result = await agent.run(goal, context or {})
            return result
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return {
                "error": f"Agent execution failed: {str(e)}",
                "agent": agent_name,
                "goal": goal
            }
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get LLM router statistics"""
        return self.llm_router.get_stats()

# Example usage
if __name__ == "__main__":
    
    async def demo_agents():
        """Demonstrate agent functionality"""
        
        # Sample LLM configuration
        llm_config = {
            'ollama': {
                'enabled': False,  # Set to True if Ollama is available
                'endpoint': 'http://localhost:11434',
                'models': ['llama3:8b', 'qwen2.5:7b']
            },
            'qwen': {
                'enabled': False,  # Set to True if Qwen models are available
                'models': ['Qwen2.5-7B-Instruct']
            },
            'openai': {
                'api_key': 'your-openai-key',
                'models': {
                    'gpt-4o-mini': {'cost': 0.00015, 'context': 128000},
                    'gpt-4': {'cost': 0.03, 'context': 8000}
                }
            }
        }
        
        # Initialize agent factory
        factory = AgentFactory(llm_config)
        
        print("Available agents:", factory.list_agents())
        
        # Demo analyst agent
        context = {
            'daily_metrics': {
                'total_revenue': 50000,
                'invoices_sent': 15,
                'payments_received': 8
            },
            'previous_metrics': {
                'total_revenue': 45000,
                'invoices_sent': 12,
                'payments_received': 10
            },
            'recent_activities': [
                {'type': 'invoice_sent', 'amount': 5000, 'client': 'Acme Corp'},
                {'type': 'payment_received', 'amount': 3000, 'client': 'Tech Inc'}
            ],
            'alerts': [
                {'type': 'overdue_invoice', 'severity': 'medium', 'count': 3}
            ]
        }
        
        result = await factory.run_agent('analyst', 'Generate daily business brief', context)
        print("\nAnalyst Agent Result:")
        print(json.dumps(result, indent=2))
        
        # Show router stats
        print("\nLLM Router Stats:")
        print(json.dumps(factory.get_router_stats(), indent=2))
    
    # Run demo
    # asyncio.run(demo_agents())