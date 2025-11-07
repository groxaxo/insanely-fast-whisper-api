#!/bin/bash

# Install Whisper API as a systemd service for autostart on boot

echo "Installing Whisper API systemd service..."

# Copy service file to systemd directory
sudo cp whisper-api.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable whisper-api.service

# Start the service now
sudo systemctl start whisper-api.service

# Check status
echo ""
echo "Service installed and started!"
echo ""
echo "Commands:"
echo "  sudo systemctl status whisper-api   # Check status"
echo "  sudo systemctl stop whisper-api     # Stop service"
echo "  sudo systemctl start whisper-api    # Start service"
echo "  sudo systemctl restart whisper-api  # Restart service"
echo "  sudo journalctl -u whisper-api -f   # View logs"
echo ""
echo "The service will now start automatically on boot!"
