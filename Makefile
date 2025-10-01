# Makefile for Scene Completion Project
# Provides convenient shortcuts for common tasks

.PHONY: help install test prepare train infer visualize clean

# Default target
help:
	@echo "Scene Completion - Available Commands:"
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make test            Run quick test with synthetic data"
	@echo "  make prepare         Prepare RealSense data"
	@echo "  make train           Train model"
	@echo "  make infer           Run inference"
	@echo "  make visualize       Visualize results"
	@echo "  make clean           Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make train DATA=data/videos EPOCHS=100"
	@echo "  make infer VIDEO=test.mp4 CHECKPOINT=outputs/best.pt"

# Installation
install:
	pip install -r requirements.txt
	mkdir -p data outputs checkpoints results logs
	@echo "Installation complete!"

# Quick test
test:
	python run.py test --epochs 10

# Prepare RealSense data
prepare:
	@if [ -z "$(INPUT)" ]; then \
		echo "Error: INPUT not specified. Usage: make prepare INPUT=camera/colorFrames OUTPUT=data/videos"; \
		exit 1; \
	fi
	python run.py prepare --input $(INPUT) --output $(or $(OUTPUT),data/videos)

# Training
train:
	python run.py train \
		--data $(or $(DATA),data/videos) \
		--epochs $(or $(EPOCHS),100) \
		--batch_size $(or $(BATCH_SIZE),4) \
		--model_type $(or $(MODEL),lightweight)

# Inference
infer:
	@if [ -z "$(VIDEO)" ] || [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: VIDEO and CHECKPOINT required. Usage: make infer VIDEO=test.mp4 CHECKPOINT=outputs/best.pt"; \
		exit 1; \
	fi
	python run.py infer \
		--video $(VIDEO) \
		--checkpoint $(CHECKPOINT) \
		--output $(or $(OUTPUT),scene.ply)

# Batch inference
infer-batch:
	@if [ -z "$(VIDEO_DIR)" ] || [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: VIDEO_DIR and CHECKPOINT required."; \
		exit 1; \
	fi
	python run.py infer \
		--video_dir $(VIDEO_DIR) \
		--checkpoint $(CHECKPOINT) \
		--output_dir $(or $(OUTPUT_DIR),results)

# Visualization
visualize:
	@if [ -z "$(MESH)" ]; then \
		echo "Error: MESH required. Usage: make visualize MESH=scene.ply"; \
		exit 1; \
	fi
	python run.py visualize --mesh $(MESH)

# Complete pipeline
pipeline:
	python run.py pipeline \
		--input $(or $(INPUT),camera/colorFrames) \
		--output $(or $(OUTPUT),results)

# Webcam demo
webcam:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT required. Usage: make webcam CHECKPOINT=outputs/best.pt"; \
		exit 1; \
	fi
	python inference.py --checkpoint $(CHECKPOINT) --webcam --display

# Clean generated files
clean:
	rm -rf outputs/*/
	rm -rf results/*
	rm -rf logs/*
	rm -rf __pycache__
	rm -rf *.pyc
	@echo "Cleaned generated files"

# Deep clean (including data)
clean-all: clean
	rm -rf data/videos/*
	rm -rf checkpoints/*
	@echo "Deep clean complete"

# Development
dev-install:
	pip install -r requirements.txt
	pip install pytest black flake8 ipython jupyter
	@echo "Dev environment ready"

# Run tests
pytest:
	pytest tests/ -v

# Format code
format:
	black *.py
	@echo "Code formatted"

# Lint
lint:
	flake8 *.py --max-line-length=100
	@echo "Linting complete"

# Quick examples
example-synthetic:
	python quick_start.py --mode train --epochs 20
	python quick_start.py --mode infer --video test.mp4 --checkpoint outputs/model.pt

example-realsense:
	make prepare INPUT=camera/colorFrames OUTPUT=data/videos
	make train DATA=data/videos EPOCHS=50
	make infer VIDEO=data/videos/color.mp4 CHECKPOINT=outputs/*/best.pt

# Docker (if using containers)
docker-build:
	docker build -t scene-completion .

docker-run:
	docker run --gpus all -it -v $(PWD):/workspace scene-completion