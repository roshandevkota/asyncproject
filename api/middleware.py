from django.utils.deprecation import MiddlewareMixin

class CustomCSPMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.path.startswith('/media/'):
            response['Content-Security-Policy'] = "frame-ancestors 'self' http://localhost:3000"
        return response
