# =========================
# MODEL
# =========================
class ParallelHybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.cnn = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.vit.num_features + self.cnn.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)
        cnn_feat = self.cnn(x)
        fused = torch.cat([vit_feat, cnn_feat], dim=1)
        return self.classifier(fused)
