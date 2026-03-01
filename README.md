# Tuna Simulator

魚群Boidsパラメータを映像を利用して調整するUnityプロジェクト

## フォルダ構成

```markdown
tunaSimulator/
├── Assets/                     # Unityプロジェクトのアセット
│   ├── Scenes/                 # シーンファイル
│   ├── Scripts/                # スクリプト
│   ├── Prefabs/                # プレハブ
│   ├── Materials/              # マテリアル
│   ├── ThirdParty/             # 外部アセット
│   └──  ...
├── ProjectSettings/
├── Packages/
├── .gitignore
├── README.md
└── ...
```

## 各アセットについて

importした外部アセットはAssets/ThirdParty内にある．

- 地形のテクスチャ
[Handpainted Grass and Ground Textures](https://assetstore.unity.com/packages/2d/textures-materials/nature/handpainted-grass-ground-textures-187634#content)

- 水のテクスチャ
[Simple Water Shader URP](https://assetstore.unity.com/packages/2d/textures-materials/water/simple-water-shader-urp-191449#content)

## 実行方法
- 以下のディレクトリ内に模倣する魚群映像データを用意する
```markdown
tunaSimulator/Assets/Scripts/server
```

- 魚群映像データを含むディレクトリを以下のコード内で指定し，実行する
```markdown
tunaSimulator/Assets/Scripts/server/server.py
```

- unityプロジェクトを起動し，以下のシーンを実行する
```markdown
tunaSimulator/Assets/Seanes/ga-simulation
```

