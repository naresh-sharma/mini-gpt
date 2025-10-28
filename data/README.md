# 📊 Data

Training datasets and data utilities for MiniGPT.

## 📁 Files

- `tiny_shakespeare.txt` - Small Shakespeare dataset for training (1MB)
- `.gitkeep` - Keeps this directory in version control

## 📥 Download Data

The main dataset used in this series is the Tiny Shakespeare dataset:

```python
# Download tiny_shakespeare.txt
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "tiny_shakespeare.txt")
```

## 📊 Dataset Information

**Tiny Shakespeare Dataset:**
- **Size**: ~1MB of text
- **Content**: Complete works of Shakespeare
- **Characters**: ~1M characters
- **Use Case**: Perfect for learning and experimentation

## 🔧 Data Processing

The notebooks will show you how to:
- Load and preprocess text data
- Create training sequences
- Handle different text formats
- Prepare data for tokenization

## 📈 Other Datasets

You can experiment with other datasets:
- **Tiny Stories**: Simple children's stories
- **Your own text**: Any text file you want to use
- **Code**: Try training on code repositories

## 💡 Tips

- Start with small datasets for learning
- Use the same dataset across all parts for consistency
- Experiment with different text sources
- Consider data quality and preprocessing

## 🤝 Contributing

Have a good dataset to share? [Submit a PR](https://github.com/naresh-sharma/mini-gpt/pulls) to add it here!
