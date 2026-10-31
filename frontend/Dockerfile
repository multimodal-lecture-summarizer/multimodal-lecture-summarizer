# Frontend image: Vite build served by nginx. Build context: ./frontend
# VITE_* values are baked in at build time; pass the URL the browser uses to reach the API.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ARG VITE_BACKEND_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_BACKEND_URL=$VITE_BACKEND_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
