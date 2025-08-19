# FastAPI Application - Main API endpoints for AI Automation Platform
# Provides REST API for all components: insights, policies, agents, review queue, vector memory, payments

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import uvicorn

# Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# Import our modules
from insights_engine import InsightsEngine, InsightsConfig
from policy_engine import PolicyEngine, PolicyDecision, ValidationResult
from agents import AgentFactory, LLMRouter
from review_queue import ReviewQueue, ReviewRequest, ReviewDecision
from vector_memory import VectorMemorySystem, MemoryItem, SearchResult, VectorConfig
from payments import PaymentProcessor, PaymentRequest, CryptoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ai_automation")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app
app = FastAPI(
    title="AI Automation Platform",
    description="Comprehensive AI automation for insights, policies, agents, review workflows, vector memory, and payments",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Static files for payment pages
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simple authentication (replace with proper auth in production)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT token verification
    if credentials.credentials != "your-api-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return credentials.credentials

# Pydantic models for API requests/responses
class InsightsResponse(BaseModel):
    kpis: Dict[str, Any]
    ar_aging: Dict[str, Any]
    trends: Dict[str, Any]
    anomalies: Dict[str, Any]
    forecast: Dict[str, Any]
    generated_at: str

class PolicyEvaluationRequest(BaseModel):
    policy_name: str
    data: Dict[str, Any]

class AgentRunRequest(BaseModel):
    agent_name: str
    goal: str
    context: Dict[str, Any] = {}

class ReviewQueueRequest(BaseModel):
    invoice_number: str
    invoice_data: Dict[str, Any]
    reason: str
    priority: str = "medium"
    confidence_score: float = 0.0
    assigned_to: Optional[str] = None

class ReviewDecisionRequest(BaseModel):
    action: str
    notes: str = ""
    changes: Dict[str, Any] = {}
    escalate: bool = False
    escalate_to: Optional[str] = None

class MemoryAddRequest(BaseModel):
    content: str
    title: str = ""
    source: str = "api"
    memory_type: str = "knowledge"
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    expires_hours: Optional[int] = None

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10
    memory_type: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    min_similarity: Optional[float] = None

class PaymentCreateRequest(BaseModel):
    invoice_number: str
    amount: float
    currency: str
    customer_email: str
    customer_name: str = ""
    payment_methods: List[str] = []
    description: str = ""
    success_url: str = ""
    cancel_url: str = ""
    expires_hours: int = 24
    metadata: Dict[str, Any] = {}

# Initialize components (would be done at startup in production)
insights_config = InsightsConfig()
vector_config = VectorConfig()
crypto_config = CryptoConfig()

# LLM configuration
llm_config = {
    'ollama': {
        'enabled': False,
        'endpoint': 'http://localhost:11434',
        'models': ['llama3:8b', 'qwen2.5:7b']
    },
    'qwen': {
        'enabled': False,
        'models': ['Qwen2.5-7B-Instruct']
    },
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'models': {
            'gpt-4o-mini': {'cost': 0.00015, 'context': 128000},
            'gpt-4': {'cost': 0.03, 'context': 8000}
        }
    }
}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting AI Automation Platform...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Automation Platform...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Insights Engine Endpoints
@app.get("/insights/summary", response_model=InsightsResponse)
async def get_insights_summary(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get comprehensive insights summary"""
    try:
        engine = InsightsEngine(db, insights_config)
        summary = engine.get_summary()
        return InsightsResponse(**summary)
    except Exception as e:
        logger.error(f"Failed to get insights summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")

@app.get("/insights/kpis")
async def get_kpis(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get key performance indicators"""
    try:
        engine = InsightsEngine(db, insights_config)
        return engine.compute_kpis()
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute KPIs")

@app.get("/insights/anomalies")
async def get_anomalies(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get detected anomalies"""
    try:
        engine = InsightsEngine(db, insights_config)
        return engine.detect_anomalies()
    except Exception as e:
        logger.error(f"Failed to get anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect anomalies")

@app.get("/insights/forecast")
async def get_forecast(
    days_ahead: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get cash flow forecast"""
    try:
        engine = InsightsEngine(db, insights_config)
        return engine.forecast_cash_flow(days_ahead)
    except Exception as e:
        logger.error(f"Failed to get forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

# Policy Engine Endpoints
@app.post("/policies/evaluate")
async def evaluate_policy(
    request: PolicyEvaluationRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Evaluate a specific policy against data"""
    try:
        engine = PolicyEngine(db)
        decision = engine.evaluate_policy(request.policy_name, request.data)
        return {
            "action": decision.action.value,
            "confidence": decision.confidence,
            "triggered_rules": decision.triggered_rules,
            "severity": decision.severity.value,
            "message": decision.message,
            "metadata": decision.metadata,
            "policy_name": decision.policy_name
        }
    except Exception as e:
        logger.error(f"Failed to evaluate policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate policy")

@app.post("/policies/validate")
async def validate_data(
    data: Dict[str, Any],
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Validate data against all policies"""
    try:
        engine = PolicyEngine(db)
        result = engine.validate_invoice_data(data)
        return {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "confidence": result.confidence,
            "suggested_fixes": result.suggested_fixes
        }
    except Exception as e:
        logger.error(f"Failed to validate data: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate data")

@app.get("/policies/routing")
async def get_routing_decision(
    data: Dict[str, Any],
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get routing decision for data"""
    try:
        engine = PolicyEngine(db)
        decision = engine.get_routing_decision(data)
        return {
            "action": decision.action.value,
            "confidence": decision.confidence,
            "triggered_rules": decision.triggered_rules,
            "severity": decision.severity.value,
            "message": decision.message,
            "metadata": decision.metadata
        }
    except Exception as e:
        logger.error(f"Failed to get routing decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to get routing decision")

@app.get("/policies")
async def list_policies(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """List all available policies"""
    try:
        engine = PolicyEngine(db)
        return engine.get_all_policies()
    except Exception as e:
        logger.error(f"Failed to list policies: {e}")
        raise HTTPException(status_code=500, detail="Failed to list policies")

# Agents Endpoints
@app.post("/agents/run")
async def run_agent(
    request: AgentRunRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Run a specific agent"""
    try:
        factory = AgentFactory(llm_config)
        result = await factory.run_agent(request.agent_name, request.goal, request.context)
        return result
    except Exception as e:
        logger.error(f"Failed to run agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to run agent")

@app.get("/agents")
async def list_agents(token: str = Depends(verify_token)):
    """List available agents"""
    try:
        factory = AgentFactory(llm_config)
        return {
            "agents": factory.list_agents(),
            "llm_stats": factory.get_router_stats()
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@app.get("/agents/stats")
async def get_agent_stats(token: str = Depends(verify_token)):
    """Get LLM router statistics"""
    try:
        factory = AgentFactory(llm_config)
        return factory.get_router_stats()
    except Exception as e:
        logger.error(f"Failed to get agent stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent stats")

# Review Queue Endpoints
@app.post("/review-queue/add")
async def add_to_review_queue(
    request: ReviewQueueRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Add item to review queue"""
    try:
        from review_queue import ReviewRequest, ReviewReason, ReviewPriority
        
        queue = ReviewQueue(db)
        review_request = ReviewRequest(
            invoice_number=request.invoice_number,
            invoice_data=request.invoice_data,
            reason=ReviewReason(request.reason),
            priority=ReviewPriority(request.priority),
            confidence_score=request.confidence_score,
            assigned_to=request.assigned_to
        )
        
        item_id = queue.add_to_queue(review_request)
        return {"item_id": item_id, "message": "Item added to review queue"}
    except Exception as e:
        logger.error(f"Failed to add to review queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to review queue")

@app.get("/review-queue/items")
async def get_review_queue_items(
    status: Optional[str] = None,
    assigned_to: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get items from review queue"""
    try:
        from review_queue import ReviewStatus, ReviewPriority
        
        queue = ReviewQueue(db)
        
        status_enum = ReviewStatus(status) if status else None
        priority_enum = ReviewPriority(priority) if priority else None
        
        items = queue.get_queue_items(status_enum, assigned_to, priority_enum, limit, offset)
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.error(f"Failed to get review queue items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review queue items")

@app.get("/review-queue/item/{item_id}")
async def get_review_item(
    item_id: int,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get specific review item"""
    try:
        queue = ReviewQueue(db)
        item = queue.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review item: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review item")

@app.post("/review-queue/item/{item_id}/review")
async def review_item(
    item_id: int,
    request: ReviewDecisionRequest,
    reviewed_by: str,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Submit review decision for item"""
    try:
        from review_queue import ReviewDecision, ReviewStatus
        
        queue = ReviewQueue(db)
        decision = ReviewDecision(
            action=ReviewStatus(request.action),
            notes=request.notes,
            changes=request.changes,
            escalate=request.escalate,
            escalate_to=request.escalate_to
        )
        
        success = queue.review_item(item_id, decision, reviewed_by)
        if not success:
            raise HTTPException(status_code=404, detail="Review item not found")
        
        return {"message": "Review decision submitted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit review decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit review decision")

@app.get("/review-queue/summary")
async def get_review_queue_summary(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get review queue summary statistics"""
    try:
        queue = ReviewQueue(db)
        summary = queue.get_summary()
        return {
            "total_pending": summary.total_pending,
            "high_priority": summary.high_priority,
            "overdue": summary.overdue,
            "avg_review_time_hours": summary.avg_review_time_hours,
            "approval_rate": summary.approval_rate,
            "by_reason": summary.by_reason,
            "by_priority": summary.by_priority
        }
    except Exception as e:
        logger.error(f"Failed to get review queue summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get review queue summary")

# Vector Memory Endpoints
@app.post("/memory/add")
async def add_memory(
    request: MemoryAddRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Add item to vector memory"""
    try:
        from vector_memory import MemoryType
        
        memory_system = VectorMemorySystem(db, vector_config)
        memory_item = MemoryItem(
            content=request.content,
            title=request.title,
            source=request.source,
            memory_type=MemoryType(request.memory_type),
            tags=request.tags,
            metadata=request.metadata,
            expires_hours=request.expires_hours
        )
        
        memory_ids = memory_system.add_memory(memory_item)
        return {"memory_ids": memory_ids, "message": f"Added {len(memory_ids)} memory chunks"}
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to add memory")

@app.post("/memory/search")
async def search_memories(
    request: MemorySearchRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Search vector memories"""
    try:
        from vector_memory import MemoryType
        
        memory_system = VectorMemorySystem(db, vector_config)
        
        memory_type_enum = MemoryType(request.memory_type) if request.memory_type else None
        
        results = memory_system.search_memories(
            query=request.query,
            limit=request.limit,
            memory_type=memory_type_enum,
            tags=request.tags,
            source=request.source,
            min_similarity=request.min_similarity
        )
        
        return {
            "results": [
                {
                    "content": result.content,
                    "title": result.title,
                    "source": result.source,
                    "similarity_score": result.similarity_score,
                    "memory_type": result.memory_type,
                    "tags": result.tags,
                    "metadata": result.metadata,
                    "access_count": result.access_count,
                    "last_accessed": result.last_accessed.isoformat() if result.last_accessed else None
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to search memories")

@app.get("/memory/context")
async def get_context(
    query: str,
    max_length: int = Query(2000, ge=100, le=10000),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get relevant context for query"""
    try:
        memory_system = VectorMemorySystem(db, vector_config)
        context = memory_system.get_context_for_query(query, max_length)
        return {"context": context, "length": len(context)}
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail="Failed to get context")

@app.get("/memory/stats")
async def get_memory_stats(
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get vector memory statistics"""
    try:
        memory_system = VectorMemorySystem(db, vector_config)
        return memory_system.get_memory_stats()
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory stats")

# Payment Endpoints
@app.post("/payments/create")
async def create_payment(
    request: PaymentCreateRequest,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Create a new payment"""
    try:
        from payments import PaymentMethod, Currency
        
        processor = PaymentProcessor(
            db, 
            stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
            crypto_config=crypto_config
        )
        
        payment_methods = [PaymentMethod(method) for method in request.payment_methods] if request.payment_methods else None
        
        payment_request = PaymentRequest(
            invoice_number=request.invoice_number,
            amount=request.amount,
            currency=Currency(request.currency.lower()),
            customer_email=request.customer_email,
            customer_name=request.customer_name,
            payment_methods=payment_methods,
            description=request.description,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
            expires_hours=request.expires_hours,
            metadata=request.metadata
        )
        
        result = await processor.create_payment(payment_request)
        
        return {
            "payment_id": result.payment_id,
            "payment_url": result.payment_url,
            "amount": result.amount,
            "currency": result.currency,
            "expires_at": result.expires_at.isoformat(),
            "payment_methods": result.payment_methods,
            "qr_codes": result.qr_codes
        }
    except Exception as e:
        logger.error(f"Failed to create payment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create payment")

@app.get("/payments/{payment_id}/status")
async def get_payment_status(
    payment_id: str,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get payment status"""
    try:
        processor = PaymentProcessor(
            db,
            stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
            crypto_config=crypto_config
        )
        
        status = await processor.check_payment_status(payment_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get payment status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get payment status")

@app.post("/payments/webhook/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Handle Stripe webhook events"""
    try:
        body = await request.body()
        event_data = await request.json()
        
        processor = PaymentProcessor(
            db,
            stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
            crypto_config=crypto_config
        )
        
        # Process webhook in background
        background_tasks.add_task(processor.handle_stripe_webhook, event_data)
        
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Failed to handle Stripe webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@app.get("/payments/analytics")
async def get_payment_analytics(
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get payment analytics"""
    try:
        processor = PaymentProcessor(
            db,
            stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
            crypto_config=crypto_config
        )
        
        return processor.get_payment_analytics(days_back)
    except Exception as e:
        logger.error(f"Failed to get payment analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get payment analytics")

@app.post("/payments/reconcile")
async def reconcile_payments(
    invoice_numbers: Optional[List[str]] = None,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Reconcile payments with invoices"""
    try:
        processor = PaymentProcessor(
            db,
            stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
            crypto_config=crypto_config
        )
        
        return processor.reconcile_payments(invoice_numbers)
    except Exception as e:
        logger.error(f"Failed to reconcile payments: {e}")
        raise HTTPException(status_code=500, detail="Failed to reconcile payments")

# Background Tasks Endpoints
@app.post("/tasks/run-insights")
async def run_insights_task(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Run insights computation task"""
    from insights_engine import run_nightly_insights
    background_tasks.add_task(run_nightly_insights, db, insights_config)
    return {"message": "Insights computation task started"}

@app.post("/tasks/run-anomaly-scan")
async def run_anomaly_scan_task(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Run anomaly detection scan"""
    from insights_engine import run_anomaly_scan
    background_tasks.add_task(run_anomaly_scan, db, insights_config)
    return {"message": "Anomaly scan task started"}

@app.post("/tasks/memory-maintenance")
async def run_memory_maintenance_task(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Run memory system maintenance"""
    from vector_memory import run_memory_maintenance
    background_tasks.add_task(run_memory_maintenance, db, vector_config)
    return {"message": "Memory maintenance task started"}

@app.post("/tasks/payment-updates")
async def run_payment_updates_task(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Run payment status updates"""
    from payments import run_payment_status_updates
    
    processor = PaymentProcessor(
        db,
        stripe_secret_key=os.getenv('STRIPE_SECRET_KEY'),
        crypto_config=crypto_config
    )
    
    background_tasks.add_task(run_payment_status_updates, db, processor)
    return {"message": "Payment status update task started"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )