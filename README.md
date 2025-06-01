# Drone Fleet Delivery Optimization System

A Python-based system for optimizing drone fleet delivery operations under various constraints.

## Features

- Dynamic drone fleet management
- Delivery point optimization
- No-fly zone handling
- Multiple optimization algorithms (A*, CSP, Genetic Algorithm)
- Interactive visualization
- Comprehensive reporting

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

Or run with a specific configuration file:
```bash
python main.py --config data.json
```

## Project Structure

- `drone.py`: Drone class and management
- `delivery.py`: Delivery point handling
- `zone.py`: No-fly zone management
- `routing.py`: Path finding algorithms
- `optimizer.py`: CSP and GA implementations
- `visualizer.py`: Visualization tools
- `main.py`: Main application entry point
- `data/`: Sample data files
- `tests/`: Test cases

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License 