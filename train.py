import argparse
import torch
import tensorflow as tf
from pytorch_model import PyTorchModel, train_pytorch, get_pytorch_dataloaders
from tensorflow_model import TensorFlowModel, train_tensorflow
import os

def main():
    parser = argparse.ArgumentParser(description='Train brain cancer classification model')
    parser.add_argument('--framework', type=str, required=True, 
                        choices=['pytorch', 'tensorflow'], 
                        help='Framework to use for training')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--model_name', type=str, default='Fabrice', 
                        help='Your first name for model filename')
    
    args = parser.parse_args()

    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)

    print("Script démarré")
    print(f"Framework selected : {args.framework}")
    print(f"Data folder : {args.data_dir}")
    print(f"Epochs : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Learning rate : {args.learning_rate}")
    print(f"Modele name : {args.model_name}")

    if args.framework == 'pytorch':
        print("🚧 Initializing PyTorch Training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using the device : {device}")

        # Charger les données et récupérer les classes
        train_loader, test_loader = get_pytorch_dataloaders(args.data_dir, args.batch_size)
        num_classes = len(train_loader.dataset.dataset.classes)
        print(f"Number of classes detected: {num_classes}")

        model = PyTorchModel(num_classes=num_classes).to(device)
        train_pytorch(model, train_loader, test_loader, args.epochs, args.learning_rate, device)

        # Sauvegarde du modèle
        torch.save(model.state_dict(), f'models/{args.model_name}_model.torch')
        print(f"PyTorch model saved in : models/{args.model_name}_model.torch")

    elif args.framework == 'tensorflow':
        print("🚧 Initializing TensorFlow Training...")

        model = TensorFlowModel()
        train_tensorflow(model, args.data_dir, args.batch_size, args.epochs, args.learning_rate)

        # Sauvegarde du modèle
        model.save(f'models/{args.model_name}_model.tensorflow')
        print(f"TensorFlow model saved in : models/{args.model_name}_model.tensorflow")

if __name__ == '__main__':
    main()
