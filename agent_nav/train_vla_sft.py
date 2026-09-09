import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from agent_nav.vla_dataset import VLADataset
from agent_nav.vla_model import build_vla


def parse_args():
    parser = argparse.ArgumentParser(description="VLA 模型 SFT 训练（OpenCLIP + StateTransformer）")
    parser.add_argument("--data-pattern", default=os.path.join("agent_nav", "data", "*.npz"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-out", default=os.path.join("agent_nav", "models", "vla_policy.pt"))
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--train-clip", action="store_true")
    parser.add_argument("--state-width", type=int, default=256)
    parser.add_argument("--fusion-width", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    dataset = VLADataset.from_pattern(args.data_pattern)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False
    )

    _, sample_state, _, _ = dataset[0]
    state_dim = int(sample_state.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vla(
        state_dim=state_dim,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        train_clip=args.train_clip,
        state_width=args.state_width,
        fusion_width=args.fusion_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for image, state, texts, action in train_loader:
            image = image.to(device)
            state = state.to(device)
            action = action.to(device)

            pred = model(image, state, texts)
            loss = F.mse_loss(pred, action)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * image.shape[0]

        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, state, texts, action in val_loader:
                image = image.to(device)
                state = state.to(device)
                action = action.to(device)
                pred = model(image, state, texts)
                loss = F.mse_loss(pred, action)
                val_loss += float(loss.item()) * image.shape[0]
        val_loss /= len(val_set)

        if val_loss < best_val:
            best_val = val_loss
            payload = {
                "state_dict": model.state_dict(),
                "state_dim": state_dim,
                "clip_model_name": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "train_clip": args.train_clip,
                "state_width": args.state_width,
                "fusion_width": args.fusion_width,
            }
            torch.save(payload, args.model_out)
        print(f"[Epoch {epoch + 1}/{args.epochs}] train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    meta_path = args.model_out + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_pattern": args.data_pattern,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "best_val_mse": best_val,
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Done] VLA 模型保存: {args.model_out}")


if __name__ == "__main__":
    main()

