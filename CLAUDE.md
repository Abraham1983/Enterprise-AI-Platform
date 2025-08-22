# Claude Code Memory - Enterprise AI Platform

## Project Owner Information
- **Name**: Abraham Vasquez
- **Email**: abraham.vasquez@enterprise-ai.com
- **LinkedIn**: [Add your LinkedIn profile URL here]
- **GitHub**: Abraham1983
- **Role**: Founder/CTO - Enterprise AI Platform

## API Keys & Services
```bash
# Add your actual API keys to .env file:
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
STRIPE_SECRET_KEY=sk_test_your-stripe-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret

# Crypto Addresses (add your actual addresses):
CRYPTO_BITCOIN_ADDRESS=your-bitcoin-address
CRYPTO_ETHEREUM_ADDRESS=your-ethereum-address
WEB3_PROVIDER_URL=your-web3-provider-url
```

## Project Details
- **Repository**: Enterprise-AI-Platform
- **Tech Stack**: FastAPI, PostgreSQL, Redis, Docker
- **AI Models**: GPT-4, Claude 3 Sonnet, GPT-3.5 Turbo, Ollama
- **Features**: Multi-model AI, Payment processing, Vector memory, Policy automation
- **Business Case**: 552% ROI through intelligent automation

## Development Commands
```bash
# Start platform
docker-compose up -d

# Run tests (when implemented)
pytest tests/

# Lint & format (when configured)
black src/
flake8 src/

# Database migrations (when using Alembic)
alembic upgrade head
```

## Common Tasks
1. **Testing API**: Use http://localhost:8000/docs
2. **View Logs**: `docker-compose logs -f api`
3. **Database Access**: Connect to postgresql://ai_automation:secure_password_2024@localhost:5432/ai_automation
4. **Monitoring**: Grafana at http://localhost:3000 (admin/admin_password_2024)

## Important Notes
- Always use environment variables for sensitive data
- Never commit API keys to repository
- Update .env.example when adding new environment variables
- All configuration is JSON-based in /config/ directory
- Production deployment requires updating security credentials

## Project Goals
Building an enterprise-grade AI automation platform that:
- Reduces manual processes by 90%
- Provides 552% ROI through intelligent automation
- Scales from startup to enterprise
- Integrates multiple AI models and payment systems
- Offers real-time business intelligence and monitoring

---
*This file is read by Claude Code to maintain context across sessions*