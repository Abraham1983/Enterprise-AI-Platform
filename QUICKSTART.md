# Enterprise AI Platform - Quick Start Guide

Get your Enterprise AI Platform running in under 10 minutes!

## 🚀 Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **PostgreSQL 12+** (or use Docker)
- **Redis 6+** (or use Docker)
- **OpenAI API Key** (required)

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Abraham1983/Enterprise-AI-Platform.git
cd Enterprise-AI-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (REQUIRED)
nano .env
```

**Minimal required configuration:**
```bash
# Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-key-here

# Database (use Docker defaults or your own)
DATABASE_URL=postgresql://enterprise_ai:secure_password_2024@localhost:5432/enterprise_ai

# Redis (use Docker defaults or your own)
REDIS_URL=redis://:secure_redis_2024@localhost:6379/0

# Security (generate secure keys)
SECRET_KEY=your-super-secret-key-minimum-32-characters-long
API_TOKEN=your-secure-api-token-for-authentication
```

### 3. Start with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 🧪 Test the Platform

### 1. Get Business Insights

```bash
curl -H "Authorization: Bearer your-api-token" \
     http://localhost:8000/insights/summary
```

### 2. Test AI Agents

```bash
curl -X POST -H "Authorization: Bearer your-api-token" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "analyst",
       "goal": "Generate business summary",
       "context": {"revenue": 100000, "customers": 500}
     }' \
     http://localhost:8000/agents/run
```

### 3. Create a Payment

```bash
curl -X POST -H "Authorization: Bearer your-api-token" \
     -H "Content-Type: application/json" \
     -d '{
       "invoice_number": "TEST-001",
       "amount": 1500.00,
       "currency": "usd",
       "customer_email": "test@example.com",
       "payment_methods": ["stripe_card", "bitcoin"]
     }' \
     http://localhost:8000/payments/create
```

## 📊 Access Dashboards

- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3000 (admin/admin_password_2024)
- **Prometheus Metrics**: http://localhost:9090

## 🔧 Manual Setup (Alternative)

If you prefer manual setup without Docker:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Database

```bash
# Create PostgreSQL database
createdb enterprise_ai

# Run migrations (if you have alembic setup)
alembic upgrade head
```

### 3. Start Services

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start worker (optional)
celery -A src.tasks worker --loglevel=info
```

## 🛡️ Security Setup

### 1. Generate Secure Keys

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Generate API_TOKEN
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Configure Authentication

Update your `.env` file:
```bash
SECRET_KEY=your-generated-secret-key
API_TOKEN=your-generated-api-token
```

### 3. Test Authentication

```bash
# This should return 401 Unauthorized
curl http://localhost:8000/insights/summary

# This should work
curl -H "Authorization: Bearer your-api-token" \
     http://localhost:8000/insights/summary
```

## 🔌 API Integration Examples

### Python Client

```python
import httpx
import asyncio

class EnterpriseAIClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    async def get_insights(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/insights/summary",
                headers=self.headers
            )
            return response.json()
    
    async def run_agent(self, agent_name: str, goal: str, context: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agents/run",
                headers=self.headers,
                json={
                    "agent_name": agent_name,
                    "goal": goal,
                    "context": context
                }
            )
            return response.json()

# Usage
async def main():
    client = EnterpriseAIClient(
        "http://localhost:8000",
        "your-api-token"
    )
    
    insights = await client.get_insights()
    print(f"Revenue: ${insights['kpis']['total_revenue']:,.2f}")
    
    agent_result = await client.run_agent(
        "analyst",
        "Analyze current performance",
        {"period": "monthly"}
    )
    print(f"Analysis: {agent_result['result']}")

asyncio.run(main())
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class EnterpriseAIClient {
    constructor(baseURL, apiToken) {
        this.client = axios.create({
            baseURL,
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
            }
        });
    }
    
    async getInsights() {
        const response = await this.client.get('/insights/summary');
        return response.data;
    }
    
    async runAgent(agentName, goal, context) {
        const response = await this.client.post('/agents/run', {
            agent_name: agentName,
            goal,
            context
        });
        return response.data;
    }
    
    async createPayment(paymentData) {
        const response = await this.client.post('/payments/create', paymentData);
        return response.data;
    }
}

// Usage
async function main() {
    const client = new EnterpriseAIClient(
        'http://localhost:8000',
        'your-api-token'
    );
    
    try {
        const insights = await client.getInsights();
        console.log(`Revenue: $${insights.kpis.total_revenue.toLocaleString()}`);
        
        const agentResult = await client.runAgent(
            'pricing_advisor',
            'Analyze pricing strategy',
            { industry: 'technology', market: 'enterprise' }
        );
        console.log('Pricing Analysis:', agentResult.result);
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

main();
```

## 🐛 Troubleshooting

### Common Issues

#### 1. API Returns 401 Unauthorized
```bash
# Check your API token in .env
grep API_TOKEN .env

# Test with correct token
curl -H "Authorization: Bearer correct-token-here" \
     http://localhost:8000/health
```

#### 2. Database Connection Error
```bash
# Check database is running
docker-compose ps postgres

# Test database connection
psql postgresql://enterprise_ai:secure_password_2024@localhost:5432/enterprise_ai -c "SELECT 1;"
```

#### 3. OpenAI API Errors
```bash
# Check your API key is valid
curl -H "Authorization: Bearer your-openai-key" \
     https://api.openai.com/v1/models

# Check API key in environment
grep OPENAI_API_KEY .env
```

#### 4. Redis Connection Error
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
redis-cli -h localhost -p 6379 -a secure_redis_2024 ping
```

### Logs and Debugging

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs postgres
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f api

# Check container status
docker-compose ps
```

### Performance Tuning

#### For Development
```bash
# Use single worker for debugging
docker-compose up api postgres redis
```

#### For Production
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📚 Next Steps

1. **Explore API Documentation**: Visit http://localhost:8000/docs
2. **Configure Business Policies**: Edit policy configurations
3. **Add Payment Methods**: Configure Stripe and crypto addresses
4. **Setup Monitoring**: Configure Grafana dashboards
5. **Scale Services**: Add more workers and database replicas

## 🤝 Getting Help

- **Documentation**: Check the full README.md
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join our Discord community
- **Professional Support**: Contact abraham.vasquez@enterprise-ai.com

## ⭐ Success!

You now have a fully functional Enterprise AI Platform running locally! 

Start building intelligent automation for your business processes.

---

**Next**: Check out the [Full Documentation](README.md) for advanced configuration and deployment options.