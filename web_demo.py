#!/usr/bin/env python3
"""
Supply Chain Alpha Web Demo

A simple Flask web interface to visualize supply chain disruption data and trading signals.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supply Chain Disruption Alpha - Live Demo</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.8;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .metric-card.warning {
            border-left-color: #e74c3c;
        }
        .metric-card.success {
            border-left-color: #27ae60;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .signal-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .signal-strong-long {
            background: #27ae60;
            color: white;
        }
        .signal-moderate-long {
            background: #f39c12;
            color: white;
        }
        .signal-neutral {
            background: #95a5a6;
            color: white;
        }
        .risk-item, .opportunity-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid;
        }
        .risk-item {
            background: #fdf2f2;
            border-left-color: #e74c3c;
        }
        .opportunity-item {
            background: #f0f9f0;
            border-left-color: #27ae60;
        }
        .chart-container {
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 20px;
        }
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin: 20px auto;
            display: block;
            transition: background 0.3s;
        }
        .refresh-btn:hover {
            background: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Supply Chain Disruption Alpha</h1>
            <p>Real-time supply chain analysis and trading signals</p>
        </div>
        
        <div class="content">
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-card {{ 'warning' if data.analysis.disruption_score > 0.6 else 'success' if data.analysis.disruption_score < 0.3 else '' }}">
                    <div class="metric-label">Disruption Score</div>
                    <div class="metric-value">{{ "%.2f"|format(data.analysis.disruption_score) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Baltic Dry Index</div>
                    <div class="metric-value">{{ data.raw_data.shipping.baltic_dry_index }}</div>
                </div>
                <div class="metric-card {{ 'warning' if data.raw_data.ports.global_metrics.average_congestion_score > 6 else 'success' }}">
                    <div class="metric-label">Port Congestion</div>
                    <div class="metric-value">{{ "%.1f"|format(data.raw_data.ports.global_metrics.average_congestion_score) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Trading Signal</div>
                    <div class="metric-value">
                        <span class="signal-badge signal-{{ data.trading_signals.overall_signal.lower().replace('_', '-') }}">
                            {{ data.trading_signals.overall_signal.replace('_', ' ') }}
                        </span>
                    </div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="chart-container">
                <div id="shipping-rates-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="port-congestion-chart"></div>
            </div>
            
            <!-- Analysis Section -->
            <div class="section">
                <h2>🔍 Disruption Analysis</h2>
                <p><strong>Overall Disruption Score:</strong> {{ "%.2f"|format(data.analysis.disruption_score) }}/1.0</p>
                
                {% if data.analysis.risk_factors %}
                <h3>⚠️ Risk Factors</h3>
                {% for risk in data.analysis.risk_factors %}
                <div class="risk-item">{{ risk }}</div>
                {% endfor %}
                {% endif %}
                
                {% if data.analysis.opportunities %}
                <h3>💡 Opportunities</h3>
                {% for opp in data.analysis.opportunities %}
                <div class="opportunity-item">{{ opp }}</div>
                {% endfor %}
                {% endif %}
            </div>
            
            <!-- Trading Signals Section -->
            <div class="section">
                <h2>📈 Trading Signals</h2>
                <p><strong>Signal:</strong> 
                    <span class="signal-badge signal-{{ data.trading_signals.overall_signal.lower().replace('_', '-') }}">
                        {{ data.trading_signals.overall_signal.replace('_', ' ') }}
                    </span>
                </p>
                <p><strong>Confidence:</strong> {{ "%.1f"|format(data.trading_signals.confidence * 100) }}%</p>
                
                {% if data.trading_signals.positions %}
                <h3>🎯 Recommended Positions</h3>
                <ul>
                {% for position in data.trading_signals.positions %}
                <li>{{ position }}</li>
                {% endfor %}
                </ul>
                {% endif %}
                
                {% if data.trading_signals.risk_management %}
                <h3>🛡️ Risk Management</h3>
                <ul>
                {% for risk in data.trading_signals.risk_management %}
                <li>{{ risk }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            </div>
            
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
            
            <div class="timestamp">
                Last updated: {{ data.demo_info.timestamp }}
            </div>
        </div>
    </div>
    
    <script>
        // Shipping Rates Chart
        var shippingData = {
            x: ['Shanghai-LA', 'Rotterdam-NY', 'Singapore-Hamburg'],
            y: [{{ data.raw_data.shipping.container_rates.shanghai_to_los_angeles }}, 
                {{ data.raw_data.shipping.container_rates.rotterdam_to_new_york }}, 
                {{ data.raw_data.shipping.container_rates.singapore_to_hamburg }}],
            type: 'bar',
            marker: {
                color: ['#3498db', '#e74c3c', '#27ae60']
            },
            name: 'Container Rates (USD)'
        };
        
        var shippingLayout = {
            title: 'Container Shipping Rates',
            xaxis: { title: 'Route' },
            yaxis: { title: 'Rate (USD)' },
            showlegend: false
        };
        
        Plotly.newPlot('shipping-rates-chart', [shippingData], shippingLayout);
        
        // Port Congestion Chart
        var portData = {
            x: ['Los Angeles', 'Shanghai', 'Rotterdam'],
            y: [{{ data.raw_data.ports.major_ports.los_angeles.waiting_vessels }},
                {{ data.raw_data.ports.major_ports.shanghai.waiting_vessels }},
                {{ data.raw_data.ports.major_ports.rotterdam.waiting_vessels }}],
            type: 'bar',
            marker: {
                color: ['#f39c12', '#e74c3c', '#27ae60']
            },
            name: 'Waiting Vessels'
        };
        
        var portLayout = {
            title: 'Port Congestion - Waiting Vessels',
            xaxis: { title: 'Port' },
            yaxis: { title: 'Number of Vessels' },
            showlegend: false
        };
        
        Plotly.newPlot('port-congestion-chart', [portData], portLayout);
    </script>
</body>
</html>
"""

def load_demo_data():
    """Load the latest demo results."""
    try:
        with open('demo_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@app.route('/')
def dashboard():
    """Main dashboard route."""
    data = load_demo_data()
    if not data:
        return "<h1>No demo data found. Please run 'python demo.py' first.</h1>"
    
    return render_template_string(HTML_TEMPLATE, data=data)

@app.route('/api/data')
def api_data():
    """API endpoint for raw data."""
    data = load_demo_data()
    if not data:
        return jsonify({'error': 'No data available'}), 404
    return jsonify(data)

@app.route('/api/refresh')
def refresh_data():
    """Refresh data by running the demo script."""
    import subprocess
    try:
        result = subprocess.run(['python', 'demo.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return jsonify({'status': 'success', 'message': 'Data refreshed successfully'})
        else:
            return jsonify({'status': 'error', 'message': result.stderr}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Refresh timeout'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Supply Chain Alpha Web Demo...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Make sure to run 'python demo.py' first to generate data\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)