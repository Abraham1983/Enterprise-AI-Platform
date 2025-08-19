# Payment Processing - Stripe and Cryptocurrency Integration
# Handles payment creation, tracking, and reconciliation for invoices

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal
import hashlib
import requests

# Database
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

# Try to import optional dependencies
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False

try:
    import web3
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class PaymentMethod(Enum):
    STRIPE_CARD = "stripe_card"
    STRIPE_ACH = "stripe_ach"
    STRIPE_WIRE = "stripe_wire"
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    USDC = "usdc"
    USDT = "usdt"

class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

class Currency(Enum):
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    BTC = "btc"
    ETH = "eth"
    USDC = "usdc"
    USDT = "usdt"

# Database Models
class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Payment identification
    payment_id = Column(String, unique=True, index=True)
    invoice_number = Column(String, index=True)
    
    # Payment details
    payment_method = Column(SQLEnum(PaymentMethod))
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    amount = Column(Float)
    currency = Column(SQLEnum(Currency))
    amount_usd = Column(Float)  # Always store USD equivalent
    
    # Payment processor details
    stripe_payment_intent_id = Column(String)
    stripe_customer_id = Column(String)
    crypto_address = Column(String)
    crypto_transaction_hash = Column(String)
    crypto_confirmations = Column(Integer, default=0)
    
    # Customer information
    customer_email = Column(String)
    customer_name = Column(String)
    billing_address = Column(JSON)
    
    # Payment URLs and references
    payment_url = Column(String)
    checkout_session_id = Column(String)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Metadata
    metadata = Column(JSON)
    webhook_events = Column(JSON)
    
    # Reconciliation
    reconciled = Column(Boolean, default=False)
    reconciled_at = Column(DateTime)

@dataclass
class PaymentRequest:
    """Request to create a new payment"""
    invoice_number: str
    amount: float
    currency: Currency
    customer_email: str
    customer_name: str = ""
    payment_methods: List[PaymentMethod] = None
    description: str = ""
    success_url: str = ""
    cancel_url: str = ""
    expires_hours: int = 24
    metadata: Dict[str, Any] = None

@dataclass
class PaymentResult:
    """Result of payment creation"""
    payment_id: str
    payment_url: str
    amount: float
    currency: str
    expires_at: datetime
    payment_methods: List[str]
    qr_codes: Dict[str, str]  # For crypto payments

@dataclass
class CryptoConfig:
    """Configuration for cryptocurrency payments"""
    bitcoin_address: str = ""
    ethereum_address: str = ""
    usdc_contract: str = "0xA0b86a33E6e9e8C7891c8b64a1d8a8B9a4e4e9c7"  # Example USDC contract
    usdt_contract: str = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # Example USDT contract
    confirmation_blocks: int = 3
    web3_provider_url: str = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
    btc_api_url: str = "https://blockstream.info/api"

class PaymentProcessor:
    """Main payment processing system supporting Stripe and cryptocurrency"""
    
    def __init__(self, db_session: Session, stripe_secret_key: str = None, crypto_config: CryptoConfig = None):
        self.db = db_session
        self.stripe_secret_key = stripe_secret_key
        self.crypto_config = crypto_config or CryptoConfig()
        
        # Initialize Stripe
        if STRIPE_AVAILABLE and stripe_secret_key:
            stripe.api_key = stripe_secret_key
            logger.info("Stripe payment processor initialized")
        else:
            logger.warning("Stripe not available or not configured")
        
        # Initialize Web3 for Ethereum-based tokens
        if WEB3_AVAILABLE and self.crypto_config.web3_provider_url:
            try:
                self.web3 = Web3(Web3.HTTPProvider(self.crypto_config.web3_provider_url))
                logger.info(f"Web3 initialized: Connected={self.web3.is_connected()}")
            except Exception as e:
                logger.error(f"Failed to initialize Web3: {e}")
                self.web3 = None
        else:
            self.web3 = None
    
    async def create_payment(self, payment_request: PaymentRequest) -> PaymentResult:
        """Create a new payment with multiple payment method options"""
        
        try:
            # Generate unique payment ID
            payment_id = self._generate_payment_id(payment_request.invoice_number)
            
            # Calculate USD equivalent
            amount_usd = await self._convert_to_usd(payment_request.amount, payment_request.currency)
            
            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(hours=payment_request.expires_hours)
            
            # Determine available payment methods
            available_methods = payment_request.payment_methods or [
                PaymentMethod.STRIPE_CARD,
                PaymentMethod.BITCOIN,
                PaymentMethod.ETHEREUM,
                PaymentMethod.USDC
            ]
            
            # Create Stripe checkout session if Stripe is available
            payment_url = ""
            checkout_session_id = ""
            
            if PaymentMethod.STRIPE_CARD in available_methods and STRIPE_AVAILABLE and self.stripe_secret_key:
                stripe_session = await self._create_stripe_session(payment_request, payment_id, expires_at)
                if stripe_session:
                    payment_url = stripe_session['url']
                    checkout_session_id = stripe_session['id']
            
            # Generate crypto payment addresses and QR codes
            qr_codes = {}
            crypto_addresses = {}
            
            for method in available_methods:
                if method in [PaymentMethod.BITCOIN, PaymentMethod.ETHEREUM, PaymentMethod.USDC, PaymentMethod.USDT]:
                    address = self._get_crypto_address(method)
                    if address:
                        crypto_addresses[method.value] = address
                        qr_codes[method.value] = self._generate_crypto_qr_data(
                            method, address, payment_request.amount, payment_request.currency
                        )
            
            # Create payment record
            payment = Payment(
                payment_id=payment_id,
                invoice_number=payment_request.invoice_number,
                payment_method=available_methods[0],  # Primary method
                amount=payment_request.amount,
                currency=payment_request.currency,
                amount_usd=amount_usd,
                customer_email=payment_request.customer_email,
                customer_name=payment_request.customer_name,
                payment_url=payment_url,
                checkout_session_id=checkout_session_id,
                expires_at=expires_at,
                metadata=payment_request.metadata or {},
                webhook_events=[]
            )
            
            self.db.add(payment)
            self.db.commit()
            
            logger.info(f"Created payment {payment_id} for invoice {payment_request.invoice_number}")
            
            return PaymentResult(
                payment_id=payment_id,
                payment_url=payment_url or self._generate_crypto_payment_page(payment_id),
                amount=payment_request.amount,
                currency=payment_request.currency.value,
                expires_at=expires_at,
                payment_methods=[method.value for method in available_methods],
                qr_codes=qr_codes
            )
            
        except Exception as e:
            logger.error(f"Failed to create payment: {e}")
            self.db.rollback()
            raise
    
    async def _create_stripe_session(self, payment_request: PaymentRequest, payment_id: str, expires_at: datetime) -> Optional[Dict[str, Any]]:
        """Create Stripe checkout session"""
        
        try:
            # Convert amount to cents for Stripe
            amount_cents = int(payment_request.amount * 100)
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': payment_request.currency.value,
                        'product_data': {
                            'name': f'Invoice {payment_request.invoice_number}',
                            'description': payment_request.description or f'Payment for invoice {payment_request.invoice_number}'
                        },
                        'unit_amount': amount_cents,
                    },
                    'quantity': 1,
                }],
                mode='payment',
                customer_email=payment_request.customer_email,
                success_url=payment_request.success_url or f'https://yoursite.com/payment/success?payment_id={payment_id}',
                cancel_url=payment_request.cancel_url or f'https://yoursite.com/payment/cancel?payment_id={payment_id}',
                expires_at=int(expires_at.timestamp()),
                metadata={
                    'payment_id': payment_id,
                    'invoice_number': payment_request.invoice_number,
                    **(payment_request.metadata or {})
                }
            )
            
            return {
                'id': session.id,
                'url': session.url,
                'payment_intent': session.payment_intent
            }
            
        except Exception as e:
            logger.error(f"Failed to create Stripe session: {e}")
            return None
    
    def _generate_payment_id(self, invoice_number: str) -> str:
        """Generate unique payment ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{invoice_number}{timestamp}{os.urandom(8).hex()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"pay_{timestamp}_{hash_suffix}"
    
    async def _convert_to_usd(self, amount: float, currency: Currency) -> float:
        """Convert amount to USD equivalent"""
        
        if currency == Currency.USD:
            return amount
        
        try:
            # Mock conversion rates - in production, use real exchange rate API
            rates = {
                Currency.EUR: 1.1,
                Currency.GBP: 1.25,
                Currency.BTC: 45000.0,
                Currency.ETH: 3000.0,
                Currency.USDC: 1.0,
                Currency.USDT: 1.0
            }
            
            rate = rates.get(currency, 1.0)
            return amount * rate
            
        except Exception as e:
            logger.error(f"Failed to convert currency: {e}")
            return amount
    
    def _get_crypto_address(self, payment_method: PaymentMethod) -> str:
        """Get cryptocurrency address for payment method"""
        
        if payment_method == PaymentMethod.BITCOIN:
            return self.crypto_config.bitcoin_address
        elif payment_method == PaymentMethod.ETHEREUM:
            return self.crypto_config.ethereum_address
        elif payment_method == PaymentMethod.USDC:
            return self.crypto_config.ethereum_address  # USDC on Ethereum
        elif payment_method == PaymentMethod.USDT:
            return self.crypto_config.ethereum_address  # USDT on Ethereum
        
        return ""
    
    def _generate_crypto_qr_data(self, payment_method: PaymentMethod, address: str, amount: float, currency: Currency) -> str:
        """Generate QR code data for cryptocurrency payment"""
        
        try:
            if payment_method == PaymentMethod.BITCOIN:
                return f"bitcoin:{address}?amount={amount}"
            elif payment_method in [PaymentMethod.ETHEREUM, PaymentMethod.USDC, PaymentMethod.USDT]:
                return f"ethereum:{address}?value={amount}"
            
            return address
            
        except Exception as e:
            logger.error(f"Failed to generate QR data: {e}")
            return address
    
    def _generate_crypto_payment_page(self, payment_id: str) -> str:
        """Generate URL for crypto payment page"""
        return f"https://yoursite.com/payment/crypto/{payment_id}"
    
    async def check_payment_status(self, payment_id: str) -> Dict[str, Any]:
        """Check status of a payment"""
        
        try:
            payment = self.db.query(Payment).filter(Payment.payment_id == payment_id).first()
            
            if not payment:
                return {"error": "Payment not found"}
            
            # Check if payment is expired
            if payment.expires_at and datetime.utcnow() > payment.expires_at:
                if payment.status == PaymentStatus.PENDING:
                    payment.status = PaymentStatus.EXPIRED
                    self.db.commit()
            
            # Update status based on payment method
            if payment.payment_method == PaymentMethod.STRIPE_CARD and payment.checkout_session_id:
                await self._update_stripe_payment_status(payment)
            elif payment.payment_method in [PaymentMethod.BITCOIN, PaymentMethod.ETHEREUM, PaymentMethod.USDC, PaymentMethod.USDT]:
                await self._update_crypto_payment_status(payment)
            
            return {
                "payment_id": payment.payment_id,
                "status": payment.status.value,
                "amount": payment.amount,
                "currency": payment.currency.value,
                "paid_at": payment.paid_at.isoformat() if payment.paid_at else None,
                "expires_at": payment.expires_at.isoformat() if payment.expires_at else None
            }
            
        except Exception as e:
            logger.error(f"Failed to check payment status: {e}")
            return {"error": "Failed to check payment status"}
    
    async def _update_stripe_payment_status(self, payment: Payment):
        """Update payment status from Stripe"""
        
        try:
            if not STRIPE_AVAILABLE or not self.stripe_secret_key:
                return
            
            session = stripe.checkout.Session.retrieve(payment.checkout_session_id)
            
            if session.payment_status == 'paid':
                payment.status = PaymentStatus.COMPLETED
                payment.paid_at = datetime.utcnow()
                payment.stripe_payment_intent_id = session.payment_intent
            elif session.status == 'expired':
                payment.status = PaymentStatus.EXPIRED
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update Stripe payment status: {e}")
    
    async def _update_crypto_payment_status(self, payment: Payment):
        """Update payment status from blockchain"""
        
        try:
            if payment.payment_method == PaymentMethod.BITCOIN:
                await self._check_bitcoin_payment(payment)
            elif payment.payment_method in [PaymentMethod.ETHEREUM, PaymentMethod.USDC, PaymentMethod.USDT]:
                await self._check_ethereum_payment(payment)
            
        except Exception as e:
            logger.error(f"Failed to update crypto payment status: {e}")
    
    async def _check_bitcoin_payment(self, payment: Payment):
        """Check Bitcoin payment status"""
        
        try:
            # This is a simplified example - in production, use proper Bitcoin API
            address = self._get_crypto_address(payment.payment_method)
            
            # Mock Bitcoin transaction check
            # In production, query Bitcoin node or block explorer API
            
            # For now, just log the check
            logger.info(f"Checking Bitcoin payment for address {address}")
            
        except Exception as e:
            logger.error(f"Failed to check Bitcoin payment: {e}")
    
    async def _check_ethereum_payment(self, payment: Payment):
        """Check Ethereum/ERC-20 payment status"""
        
        try:
            if not self.web3 or not self.web3.is_connected():
                return
            
            address = self._get_crypto_address(payment.payment_method)
            
            if payment.payment_method == PaymentMethod.ETHEREUM:
                # Check ETH balance
                balance = self.web3.eth.get_balance(address)
                balance_eth = self.web3.from_wei(balance, 'ether')
                
                # Simple check - in production, track specific transactions
                if balance_eth >= payment.amount:
                    payment.status = PaymentStatus.COMPLETED
                    payment.paid_at = datetime.utcnow()
                    payment.crypto_confirmations = 12  # Assume confirmed
            
            elif payment.payment_method in [PaymentMethod.USDC, PaymentMethod.USDT]:
                # Check ERC-20 token balance
                # This would require contract interaction
                logger.info(f"Checking {payment.payment_method.value} payment for address {address}")
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to check Ethereum payment: {e}")
    
    def handle_stripe_webhook(self, event_data: Dict[str, Any]) -> bool:
        """Handle Stripe webhook events"""
        
        try:
            event_type = event_data.get('type')
            
            if event_type == 'checkout.session.completed':
                session = event_data['data']['object']
                payment_id = session['metadata'].get('payment_id')
                
                if payment_id:
                    payment = self.db.query(Payment).filter(Payment.payment_id == payment_id).first()
                    
                    if payment:
                        payment.status = PaymentStatus.COMPLETED
                        payment.paid_at = datetime.utcnow()
                        payment.stripe_payment_intent_id = session.get('payment_intent')
                        
                        # Add webhook event to history
                        events = payment.webhook_events or []
                        events.append({
                            'type': event_type,
                            'timestamp': datetime.utcnow().isoformat(),
                            'data': session
                        })
                        payment.webhook_events = events
                        
                        self.db.commit()
                        
                        logger.info(f"Payment {payment_id} completed via Stripe webhook")
                        return True
            
            elif event_type == 'payment_intent.payment_failed':
                payment_intent = event_data['data']['object']
                
                # Find payment by payment intent ID
                payment = self.db.query(Payment).filter(
                    Payment.stripe_payment_intent_id == payment_intent['id']
                ).first()
                
                if payment:
                    payment.status = PaymentStatus.FAILED
                    self.db.commit()
                    
                    logger.info(f"Payment {payment.payment_id} failed via Stripe webhook")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to handle Stripe webhook: {e}")
            return False
    
    def get_payment_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get payment analytics and statistics"""
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            
            payments = self.db.query(Payment).filter(
                Payment.created_at >= start_date
            ).all()
            
            if not payments:
                return {
                    "total_payments": 0,
                    "total_amount_usd": 0,
                    "success_rate": 0,
                    "by_method": {},
                    "by_status": {},
                    "by_currency": {}
                }
            
            # Basic statistics
            total_payments = len(payments)
            total_amount_usd = sum(p.amount_usd for p in payments if p.amount_usd)
            successful_payments = len([p for p in payments if p.status == PaymentStatus.COMPLETED])
            success_rate = successful_payments / total_payments if total_payments > 0 else 0
            
            # By payment method
            by_method = {}
            for payment in payments:
                method = payment.payment_method.value
                if method not in by_method:
                    by_method[method] = {"count": 0, "amount_usd": 0}
                by_method[method]["count"] += 1
                by_method[method]["amount_usd"] += payment.amount_usd or 0
            
            # By status
            by_status = {}
            for payment in payments:
                status = payment.status.value
                if status not in by_status:
                    by_status[status] = {"count": 0, "amount_usd": 0}
                by_status[status]["count"] += 1
                by_status[status]["amount_usd"] += payment.amount_usd or 0
            
            # By currency
            by_currency = {}
            for payment in payments:
                currency = payment.currency.value
                if currency not in by_currency:
                    by_currency[currency] = {"count": 0, "amount_usd": 0}
                by_currency[currency]["count"] += 1
                by_currency[currency]["amount_usd"] += payment.amount_usd or 0
            
            return {
                "total_payments": total_payments,
                "total_amount_usd": total_amount_usd,
                "success_rate": success_rate,
                "successful_payments": successful_payments,
                "by_method": by_method,
                "by_status": by_status,
                "by_currency": by_currency,
                "period_days": days_back,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get payment analytics: {e}")
            return {}
    
    def reconcile_payments(self, invoice_numbers: List[str] = None) -> Dict[str, Any]:
        """Reconcile payments with invoices"""
        
        try:
            query = self.db.query(Payment).filter(
                Payment.status == PaymentStatus.COMPLETED,
                Payment.reconciled == False
            )
            
            if invoice_numbers:
                query = query.filter(Payment.invoice_number.in_(invoice_numbers))
            
            unreconciled_payments = query.all()
            
            reconciled_count = 0
            reconciled_amount = 0
            
            for payment in unreconciled_payments:
                # Mark as reconciled
                payment.reconciled = True
                payment.reconciled_at = datetime.utcnow()
                reconciled_count += 1
                reconciled_amount += payment.amount_usd or 0
            
            self.db.commit()
            
            logger.info(f"Reconciled {reconciled_count} payments totaling ${reconciled_amount:.2f}")
            
            return {
                "reconciled_count": reconciled_count,
                "reconciled_amount_usd": reconciled_amount,
                "reconciled_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to reconcile payments: {e}")
            self.db.rollback()
            return {"error": "Reconciliation failed"}

# Helper functions
def create_payment_request(invoice_number: str, amount: float, currency: str, customer_email: str, **kwargs) -> PaymentRequest:
    """Helper to create payment request"""
    return PaymentRequest(
        invoice_number=invoice_number,
        amount=amount,
        currency=Currency(currency.lower()),
        customer_email=customer_email,
        **kwargs
    )

# Background job for payment status updates
async def run_payment_status_updates(db_session: Session, payment_processor: PaymentProcessor):
    """Background job to update payment statuses"""
    
    logger.info("Starting payment status updates")
    
    try:
        # Get pending payments
        pending_payments = db_session.query(Payment).filter(
            Payment.status.in_([PaymentStatus.PENDING, PaymentStatus.PROCESSING])
        ).all()
        
        for payment in pending_payments:
            try:
                await payment_processor.check_payment_status(payment.payment_id)
            except Exception as e:
                logger.error(f"Failed to update payment {payment.payment_id}: {e}")
        
        logger.info(f"Updated status for {len(pending_payments)} payments")
        
    except Exception as e:
        logger.error(f"Payment status update job failed: {e}")

# Example usage
if __name__ == "__main__":
    print("Payment processing system initialized with Stripe and cryptocurrency support!")