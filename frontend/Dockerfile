# Base image
FROM nginx:alpine

# Remove default Nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy static files to Nginx root directory
COPY frontend/static/ /usr/share/nginx/html/

# Copy the global Nginx configuration
COPY frontend/static/nginx.conf /etc/nginx/nginx.conf

# Copy the server configuration (default.conf)
COPY frontend/static/default.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for HTTP traffic
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
