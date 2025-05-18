FROM debian:bookworm-slim

WORKDIR /app

# Install any system dependencies your binary needs (e.g., libstdc++, libgcc, libpython if needed)
RUN apt-get update && apt-get install -y libstdc++6 libgcc-s1 && rm -rf /var/lib/apt/lists/*

# Copy the compiled Nuitka binary and .env file
COPY app .
COPY .env .

# Expose the port your app uses
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Run the compiled binary
CMD ["./app"]