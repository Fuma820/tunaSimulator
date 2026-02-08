# Tuna Simulator

魚群Boidsパラメータを映像を利用して調整するUnityプロジェクト

## フォルダ構成

```markdown
multi-auv-aquaculture-sim/
├── Assets/                     # Unityプロジェクトのアセット
│   ├── Scenes/                 # シーンファイル
│   ├── Scripts/                # スクリプト
│   ├── Prefabs/                # プレハブ
│   ├── Materials/              # マテリアル
│   ├── ThirdParty/             # 外部アセット
│   ├── Terrain/
│   ├── ProBuilder Data/
│   └──  ...
├── ProjectSettings/
├── Packages/
├── .gitignore
├── README.md
└── ...
```

## 各アセットについて

importした外部アセットはAssets/ThirdParty内にある．

### Handpainted_Grass_and_Ground_Textures

地形のテクスチャ
[Handpainted Grass and Ground Textures](https://assetstore.unity.com/packages/2d/textures-materials/nature/handpainted-grass-ground-textures-187634#content)

### Water saface

水のテクスチャ
[Simple Water Shader URP](https://assetstore.unity.com/packages/2d/textures-materials/water/simple-water-shader-urp-191449#content)

### import元が不明なアセット

- Ground textures pack
- Tree
- UnityTechnologies

## 各Prefabについて

### net-01

マグロの生簀のパーツ
作成者: 松本

### マグロ

キハダマグロのフリー素材
[https://booth.pm/ja/items/1107229](https://booth.pm/ja/items/1107229)

## Scripts

```markdown
multi-auv-aquaculture-sim/Assets/Scripts
├── particle1.cs        # 塵パーティクルを表示する機能
├── particles2.cs       # 泡パーティクルを表示する機能
└── parameter-01.csv    # パラーメータフィッティングした水のRGB値
```
