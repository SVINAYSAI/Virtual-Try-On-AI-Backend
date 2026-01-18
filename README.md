# Virtual try on AI Backend 🎨👗

A powerful and versatile AI-powered backend API for fashion and styling applications. Built with **FastAPI** and **Google Vertex AI**, this platform provides advanced image processing and virtual try-on capabilities using cutting-edge generative AI models.

## ✨ Features

### 👕 Virtual Try-On Services
- **Full Garment Virtual Try-On** - Realistic virtual fitting of complete outfits on user models
- **Two-Piece Try-On** - Separate top and bottom garment virtual fitting
- **Accessory Virtual Try-On** - Add accessories (jewelry, bags, hats, etc.) to images
- **Makeup Try-On** - Apply makeup styles and shades to face images
- **360° Model Viewer** - Generate 360-degree model rotations for immersive visualization

### 🎭 Image Processing & Transformation
- **Background Removal** - Remove and replace image backgrounds using advanced segmentation
- **Background Change** - Change image backgrounds to custom scenes
- **Image Transformation** - Apply various image filters and effects (Pilgram filters)
- **Human Model Generation** - Generate realistic human models from descriptions

### 💡 Style & Fashion Intelligence
- **Occasion-Based Styling** - Get personalized outfit recommendations based on occasions
- **Body Size Recommendations** - Receive size recommendations based on body measurements
- **Image Filtering** - Apply aesthetic filters to fashion images

## 🏗️ Project Structure

```
/
├── main.py                              # FastAPI application entry point
├── config.py                            # Vertex AI configuration & initialization
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Docker container configuration
├── __init__.py                          # Package initialization
│
├── gemini_API/                          # AI-powered endpoints using Vertex AI
│   ├── gemini_virtuval_try_on_api.py   # Full garment virtual try-on
│   ├── gemini_two_piece_tryon_api.py   # Two-piece try-on
│   ├── human_model_api.py               # Human model generation
│   ├── model_360_api.py                 # 360° model generation
│   ├── accessory_tryon_api.py           # Accessory try-on
│   ├── makeup_tryon_api.py              # Makeup try-on
│   ├── transform_image_api.py           # Image transformation
│   ├── body_size_recommendation_api.py  # Size recommendations
│   └── Occasion_Based_Styling.py        # Occasion-based recommendations
│
├── image_apis/                          # Image processing endpoints
│   ├── remove_bg_api.py                 # Background removal
│   └── change_background.py             # Background replacement
│
├── filter_API/                          # Image filtering endpoints
│   └── filter_api.py                    # Filter application
│
└── outputs/                             # Generated images output directory
    └── *.png                            # Generated model and try-on images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Cloud Project with Vertex AI enabled
- Service Account credentials (JSON file)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google Cloud credentials**
   - Place your Google Cloud Service Account JSON file in the project root
   - Update the filename in `config.py` if needed:
     ```python
     CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "your-credentials.json")
     ```

5. **Run the application**
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`

6. **Access API documentation**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t virtual-try-on-ai:latest .
```

### Run Docker Container
```bash
docker run -p 8000:8000 \
  -e PROJECT_ID=your-project-id \
  virtual-try-on-ai:latest
```

### Docker Compose (Optional)
```bash
docker-compose up --build
```

## 📡 API Endpoints

### Image Processing
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/remove-bg` | POST | Remove background from image |
| `/api/change-background` | POST | Replace image background |
| `/api/filter` | POST | Apply filters to image |

### Virtual Try-On (Gemini AI)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gemini/virtual-tryon` | POST | Full garment try-on |
| `/api/gemini/two-piece-tryon` | POST | Two-piece try-on |
| `/api/gemini/accessory-tryon` | POST | Accessory try-on |
| `/api/gemini/makeup-tryon` | POST | Makeup try-on |
| `/api/gemini/model-360` | POST | 360° model generation |
| `/api/gemini/transform-image` | POST | Image transformation |

### Model & Recommendations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gemini/generate-model` | POST | Generate human model |
| `/api/gemini/body-size` | POST | Get size recommendations |
| `/api/gemini/styling-suggestions` | POST | Get style recommendations |

## ⚙️ Configuration

### Environment Variables
```bash
PROJECT_ID=marine-set-447307-k6
LOCATION=us-central1
CREDENTIALS_PATH=./marine-set-447307-k6-d2077b260b04.json
```

### Model Configuration
The application uses **Gemini 2.5 Flash Image** model from Google Vertex AI for:
- Advanced image understanding
- Virtual try-on generation
- Model creation and transformation
- Image analysis and recommendations

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | Latest | Web framework |
| uvicorn | Latest | ASGI server |
| vertexai | Latest | Google AI integration |
| google-auth | Latest | Authentication |
| pillow | Latest | Image processing |
| rembg | Latest | Background removal |
| opencv-python-headless | Latest | Computer vision |
| mediapipe | 0.10.14 | Pose & hand detection |
| onnxruntime | Latest | Model inference |
| scikit-learn | Latest | ML utilities |
| pilgram | Latest | Image filters |

## 🔐 Security

- **CORS Enabled**: All origins allowed (configure for production)
- **Service Account**: Uses Google Cloud service account authentication
- **Credentials**: Store credentials securely, never commit to version control
- **Rate Limiting**: Built-in rate limiting for model generation (5-second intervals)

### Production Security Checklist
- [ ] Restrict CORS origins to specific domains
- [ ] Use environment variables for credentials
- [ ] Implement API authentication (JWT tokens)
- [ ] Add request validation and sanitization
- [ ] Enable HTTPS
- [ ] Set up monitoring and logging
- [ ] Implement rate limiting per user/IP

## 🧪 Testing

### Manual Testing
Use the interactive Swagger UI at `/docs` to test endpoints directly.

### Sample Request
```bash
curl -X POST "http://localhost:8000/api/remove-bg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

## 📊 Model Performance Notes

- **Background Removal**: Optimized for fashion products, handles complex backgrounds
- **Virtual Try-On**: Best results with clear, front-facing human images
- **Model Generation**: Supports various body types and skin tones
- **360° Generation**: Creates smooth rotations with consistent styling

## 🚨 Error Handling

The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `422`: Validation error
- `500`: Server error
- `503`: Service unavailable (rate limited)

## 📈 Performance Optimization

### Current Optimizations
- Lazy loading of Vertex AI models
- Image compression before processing
- Async request handling with FastAPI
- Docker optimization with minimal dependencies

### Recommendations for Scale
- Implement caching (Redis) for repeated requests
- Use message queues (Celery) for long-running tasks
- Add load balancing for multiple instances
- Implement database for request history
- Use CDN for static assets

## 🐛 Troubleshooting

### Vertex AI Initialization Fails
```
❌ Error: Credentials file not found
```
**Solution**: Ensure the service account JSON file is in the project root directory.

### Rate Limit Errors (503)
```
Service temporarily unavailable
```
**Solution**: Implement exponential backoff in client code. Default retry interval is 10 seconds.

### OutOfMemory Errors
**Solution**: Process images in smaller batches or upgrade server resources.

## 📚 Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [Gemini API](https://ai.google.dev/)
- [MediaPipe](https://mediapipe.dev/)

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit changes (`git commit -m 'Add amazing feature'`)
3. Push to branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📝 Best Practices

- **Input Validation**: Always validate image format and size
- **Error Messages**: Provide clear, actionable error responses
- **Logging**: Log all API calls for debugging and monitoring
- **Resource Cleanup**: Ensure temporary files are cleaned up
- **Documentation**: Keep endpoint documentation updated

## 📄 License

This project is proprietary and confidential.

## 👥 Support

For issues and feature requests, please contact the development team.

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
