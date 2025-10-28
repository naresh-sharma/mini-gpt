# 💡 Examples

Example scripts and use cases for MiniGPT.

## 📁 Contents

This directory contains practical examples showing how to use MiniGPT:

- `basic_usage.py` - Simple example of using MiniGPT
- `custom_training.py` - Custom training example
- `text_generation.py` - Text generation examples
- `fine_tuning.py` - Fine-tuning on custom data
- `deployment.py` - Simple deployment example

## 🚀 Quick Examples

### Basic Usage

```python
from mini_gpt import MiniGPT, SimpleTokenizer

# Load a pre-trained model (when available)
model = MiniGPT.load_pretrained("mini-gpt-small")
tokenizer = SimpleTokenizer.load("tokenizer.json")

# Generate text
prompt = "The future of AI is"
generated = model.generate(prompt, max_length=50)
print(generated)
```

### Custom Training

```python
from mini_gpt import MiniGPT, Trainer, TextDataset

# Create model
model = MiniGPT(vocab_size=10000, d_model=512)

# Prepare data
dataset = TextDataset("your_text.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=32)

# Train
trainer = Trainer(model)
trainer.train(dataloader, num_epochs=10)
```

## 📚 Learning Path

1. **Start Here**: `basic_usage.py` - Get familiar with the API
2. **Training**: `custom_training.py` - Learn to train your own model
3. **Generation**: `text_generation.py` - Explore different generation strategies
4. **Advanced**: `fine_tuning.py` - Fine-tune on specific domains
5. **Deployment**: `deployment.py` - Deploy your model

## 🎯 Use Cases

- **Text Completion**: Complete sentences and paragraphs
- **Creative Writing**: Generate stories, poems, and creative content
- **Code Generation**: Generate code snippets and functions
- **Question Answering**: Answer questions based on context
- **Translation**: Translate between languages (with training)

## 🔧 Customization

Each example shows how to:
- Modify model parameters
- Use different datasets
- Adjust training settings
- Experiment with generation parameters
- Handle different text formats

## 📊 Performance Tips

- Start with small models for experimentation
- Use appropriate batch sizes for your hardware
- Monitor training progress and adjust learning rates
- Experiment with different generation strategies
- Consider using pre-trained models as starting points

## 🤝 Contributing

Have a cool example to share?
1. Create your example script
2. Add clear documentation
3. Include sample output
4. [Submit a PR](https://github.com/naresh-sharma/mini-gpt/pulls)

## 📞 Support

Need help with an example? Check out our [FAQ](../docs/FAQ.md) or [open an issue](https://github.com/naresh-sharma/mini-gpt/issues).
