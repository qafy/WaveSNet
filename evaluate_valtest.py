from utils.evaluate import *
from utils.summaries import TensorboardSummary

if __name__ == '__main__':
    save = True
      
    save_root = './inference'
    summary = TensorboardSummary(directory="./images")
    write = summary.create_summary()
    evaluate(save = save, saved_root = save_root)
    # summary.visualize_image(writer, "pascal", )