# 🚀 Deploying Belgian Document Deidentification System on Render

This guide walks you through deploying the Belgian Document Deidentification System on Render platform.

## 📋 Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Basic Understanding**: Familiarity with environment variables and web services

## 🏗️ Deployment Architecture

The system deploys as:
- **Web Service**: FastAPI application (Main API)
- **PostgreSQL Database**: For data storage and audit logs
- **Redis Cache**: For session management and caching (optional)

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

Ensure your repository contains these Render-specific files:
- ✅ `render.yaml` - Render service configuration
- ✅ `Procfile` - Process definition
- ✅ `runtime.txt` - Python version specification
- ✅ `build.sh` - Build script
- ✅ `requirements.txt` - Optimized dependencies
- ✅ `config/render.yaml` - Render-specific configuration

### Step 2: Create Services on Render

#### Option A: Using render.yaml (Recommended)

1. **Connect Repository**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select the repository containing your code

2. **Deploy from Blueprint**:
   - Render will automatically detect `render.yaml`
   - Review the services to be created:
     - `belgian-deidentification-api` (Web Service)
     - `belgian-deidentification-db` (PostgreSQL)
     - `belgian-deidentification-redis` (Redis)
   - Click "Apply" to create all services

#### Option B: Manual Service Creation

If you prefer manual setup:

1. **Create PostgreSQL Database**:
   - Click "New" → "PostgreSQL"
   - Name: `belgian-deidentification-db`
   - Plan: Starter (free tier)
   - Database Name: `belgian_deidentification`
   - User: `belgian_deident_user`

2. **Create Web Service**:
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Name: `belgian-deidentification-api`
   - Environment: Python 3
   - Build Command: `./build.sh`
   - Start Command: `python -m belgian_deidentification.api.render_main`

### Step 3: Configure Environment Variables

Set these environment variables in your web service:

#### Required Variables
```bash
# API Configuration
BELGIAN_DEIDENT_API__HOST=0.0.0.0
BELGIAN_DEIDENT_API__PORT=10000

# Application Settings
BELGIAN_DEIDENT_DEBUG=false
BELGIAN_DEIDENT_LOG_LEVEL=INFO

# Database (auto-configured if using render.yaml)
BELGIAN_DEIDENT_DATABASE__URL=$DATABASE_URL

# File Paths
BELGIAN_DEIDENT_DATA_DIR=/opt/render/project/src/data
BELGIAN_DEIDENT_MODELS_DIR=/opt/render/project/src/data/models
BELGIAN_DEIDENT_TEMP_DIR=/tmp/belgian_deidentification

# NLP Configuration
BELGIAN_DEIDENT_NLP__USE_GPU=false
BELGIAN_DEIDENT_NLP__BATCH_SIZE=8
BELGIAN_DEIDENT_NLP__MAX_LENGTH=256
BELGIAN_DEIDENT_NLP__CONFIDENCE_THRESHOLD=0.80

# Deidentification Settings
BELGIAN_DEIDENT_DEIDENTIFICATION__MODE=anonymization
BELGIAN_DEIDENT_DEIDENTIFICATION__PRESERVE_STRUCTURE=true

# Security
BELGIAN_DEIDENT_SECURITY__ENCRYPTION_ENABLED=true
BELGIAN_DEIDENT_SECURITY__AUDIT_LOGGING=true

# Quality Assurance
BELGIAN_DEIDENT_QUALITY_ASSURANCE__ENABLE_VALIDATION=true
BELGIAN_DEIDENT_QUALITY_ASSURANCE__CONFIDENCE_THRESHOLD=0.80
```

#### Optional Variables
```bash
# Redis (if using Redis service)
BELGIAN_DEIDENT_REDIS_URL=$REDIS_URL

# Custom API Keys (if needed)
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
```

### Step 4: Deploy and Monitor

1. **Trigger Deployment**:
   - Push changes to your main branch
   - Render will automatically build and deploy
   - Monitor the build logs for any issues

2. **Verify Deployment**:
   - Check the service URL (provided by Render)
   - Visit `/health` endpoint to verify the service is running
   - Visit `/docs` for the API documentation

3. **Test the API**:
   ```bash
   curl https://your-service-url.onrender.com/health
   ```

## 🔧 Configuration Options

### Service Plans

#### Starter Plan (Free)
- **Web Service**: 512 MB RAM, 0.1 CPU
- **Database**: 1 GB storage, 97 connections
- **Limitations**: Spins down after 15 minutes of inactivity

#### Standard Plan ($7/month)
- **Web Service**: 2 GB RAM, 1 CPU
- **Database**: 10 GB storage, 97 connections
- **Benefits**: No spin-down, better performance

### Performance Optimization

For better performance on Render:

1. **Use Smaller Models**:
   ```yaml
   nlp:
     model: "DTAI-KULeuven/robbert-2023-dutch-base"  # Instead of large
   ```

2. **Optimize Batch Size**:
   ```yaml
   nlp:
     batch_size: 4  # Smaller for memory constraints
   ```

3. **Enable Caching**:
   ```yaml
   performance:
     enable_caching: true
     model_cache_size: 50
   ```

## 🔍 Monitoring and Troubleshooting

### Health Checks

Render automatically monitors these endpoints:
- `/health` - Basic health check
- `/ready` - Readiness check
- `/metrics` - Performance metrics

### Common Issues

#### 1. Build Failures
```bash
# Check build logs for:
- Missing dependencies
- Memory issues during model download
- Permission errors
```

#### 2. Memory Issues
```bash
# Solutions:
- Reduce batch_size in configuration
- Use smaller model variants
- Upgrade to Standard plan
```

#### 3. Cold Start Delays
```bash
# Solutions:
- Keep service warm with health checks
- Pre-download models in build script
- Use Redis for caching
```

### Logs and Debugging

Access logs through:
1. **Render Dashboard**: Service → Logs tab
2. **Real-time Logs**: Use Render CLI
3. **Application Logs**: Structured logging with timestamps

## 🔐 Security Considerations

### Environment Variables
- Never commit secrets to repository
- Use Render's environment variable management
- Rotate API keys regularly

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular backups (automatic on Render)

### API Security
- Implement rate limiting
- Use HTTPS (automatic on Render)
- Validate all inputs

## 📊 Scaling and Performance

### Horizontal Scaling
```yaml
# In render.yaml
services:
  - type: web
    name: belgian-deidentification-api
    plan: standard  # Required for scaling
    numInstances: 2  # Multiple instances
```

### Database Scaling
- Monitor connection usage
- Upgrade plan when needed
- Consider read replicas for heavy workloads

## 🔄 CI/CD Integration

### Automatic Deployments
Render automatically deploys when you:
1. Push to main branch
2. Merge pull requests
3. Create releases

### Custom Deploy Hooks
```yaml
# In render.yaml
services:
  - type: web
    buildFilter:
      paths:
      - src/**
      - requirements.txt
      - config/**
```

## 📞 Support and Resources

### Render Resources
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [Render Status](https://status.render.com)

### Application Support
- Check application logs first
- Review configuration settings
- Test locally with same environment variables

## 🎯 Next Steps

After successful deployment:

1. **Set up monitoring** with health checks
2. **Configure custom domain** (if needed)
3. **Set up automated backups**
4. **Implement proper logging and alerting**
5. **Scale based on usage patterns**

## 💡 Tips for Success

1. **Start Small**: Use Starter plan initially, upgrade as needed
2. **Monitor Resources**: Watch memory and CPU usage
3. **Optimize Models**: Use appropriate model sizes for your plan
4. **Cache Effectively**: Implement caching for better performance
5. **Test Thoroughly**: Test all endpoints after deployment

---

🎉 **Congratulations!** Your Belgian Document Deidentification System is now running on Render!

For additional support or questions, refer to the main README.md or create an issue in the repository.

