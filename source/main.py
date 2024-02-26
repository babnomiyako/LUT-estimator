# %% [markdown]
# ### import

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", font='MS Gothic', palette='tab10')
import numpy as np
import itertools
import pprint
import colorsys
import cv2
import os
from scipy.interpolate import RegularGridInterpolator
BASEDIR = "../"

def RGBStep(IMGSIZE, BLOCKSIZE):
    step = 0
    while step**3 < (IMGSIZE**2 // BLOCKSIZE**2):  # 色数 < 最大敷詰めタイル数
        if (step + 1)**3 > (IMGSIZE**2 // BLOCKSIZE**2):
            break
        step += 1
    return step

def Padding(img, targetX, targetY):
    # img.shapeがtargetになる様に右/下 方向に0を追加
    x, y, _ = img.shape
    # iとjがxとy以上であるかチェック
    if targetX < x or targetY < y:
        # そうでなければ元のimgを返す
        return img
    else:
        # パディングする量を計算
        pad_width = ((0, targetX - x), (0, targetY - y), (0, 0))
        img2 = np.pad(img,
                      pad_width=pad_width,
                      mode="constant",  # パディング方式は定数（constant）で値は0にする
                      constant_values=0)
        return img2

def Scale(image, scale, interpolation=cv2.INTER_NEAREST):
    if scale == 1:
        return image
    else:
        return cv2.resize(image, None, None, scale, scale, interpolation=interpolation)

class GridPlot:
    def __init__(self, gridsize: tuple, figsize: tuple = (6, 4)):
        self.fig = plt.figure(figsize=figsize)
        self.x, self.y = gridsize

    def AddImg(self, image, i: int, title="", interpolation="none"):
        ax = self.fig.add_subplot(self.x, self.y, i)

        if len(image.shape) != 3:
            raise TypeError("image shape is wrong!")
        elif image.shape[2] == 1:
            if image.dtype != np.dtype("uint8") and image.dtype != np.dtype("float32"):
                raise TypeError("image dtype is wrong!")
            elif image.dtype == np.dtype("uint8"):
                im = ax.imshow(image, cmap='gray', vmin=0, vmax=255,
                               interpolation=interpolation)
            elif image.dtype == np.dtype("float32"):
                im = ax.imshow(image, cmap='gray', vmin=0, vmax=1,
                               interpolation=interpolation)
            plt.colorbar(im, ax=ax)  # 凡例バーを表示

        elif image.shape[2] == 3:
            if image.dtype != np.dtype("uint8") and image.dtype != np.dtype("float32"):
                raise TypeError("image dtype is wrong!")
            elif image.dtype == np.dtype("uint8"):
                # print("uint8 RGB")
                ax.imshow(image, vmin=0, vmax=255, interpolation=interpolation)
            elif image.dtype == np.dtype("float32"):
                # print("float32 RGB")
                ax.imshow(image, vmin=0, vmax=1, interpolation=interpolation)

        else:
            raise TypeError("image shape is wrong!")

        ax.set_title(title)

    def Show(self):
        plt.show()

def Show(image, title="", interpolation="none"):
    fig, ax = plt.subplots()

    if len(image.shape) != 3:
        raise TypeError("image shape is wrong!")
    elif image.shape[2] == 1:
        if image.dtype != np.dtype("uint8") and image.dtype != np.dtype("float32"):
            raise TypeError("image dtype is wrong!")
        elif image.dtype == np.dtype("uint8"):
            im = ax.imshow(image, cmap='gray', vmin=0, vmax=255,
                           interpolation=interpolation)
        elif image.dtype == np.dtype("float32"):
            im = ax.imshow(image, cmap='gray', vmin=0, vmax=1,
                           interpolation=interpolation)
        plt.colorbar(im, ax=ax)  # 凡例バーを表示

    elif image.shape[2] == 3:
        if image.dtype != np.dtype("uint8") and image.dtype != np.dtype("float32"):
            raise TypeError("image dtype is wrong!")
        elif image.dtype == np.dtype("uint8"):
            # print("uint8 RGB")
            ax.imshow(image, vmin=0, vmax=255, interpolation=interpolation)
        elif image.dtype == np.dtype("float32"):
            # print("float32 RGB")
            ax.imshow(image, vmin=0, vmax=1, interpolation=interpolation)

    else:
        raise TypeError("image shape is wrong!")

    ax.set_title(title)
    plt.show()


def Save(image, filename, scale=1):  # uint8のみ
    if image.dtype != np.dtype("uint8"):
        image = np.clip(image * 255, a_min=0, a_max=255).astype("uint8")

    image_scaled = Scale(image, scale)
    cv2.imwrite(filename, cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB))

def Read(filename, scale=1):
    return Scale(
        cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), scale
    )

def Devide4(img):
    """
    Divide an RGB image into 4 equally sized quadrants and return a list of 4 numpy arrays.

    Args:
    img: a numpy array of shape (2x, 2y, 3)

    Returns:
    a list of 4 numpy arrays of shape (x, y, 3)
    """
    # Calculate the dimensions of each quadrant
    x, y = img.shape[0] // 2, img.shape[1] // 2

    # Divide the image into quadrants
    img1 = img[:x, :y, :]
    img2 = img[:x, y:, :]
    img3 = img[x:, :y, :]
    img4 = img[x:, y:, :]

    return (img1, img2, img3, img4)

def Merge4(img_list):
    """
    Combine 4 RGB images of shape (x, y, 3) into a single image of shape (2x, 2y, 3).

    Args:
    img_list: a list of 4 numpy arrays of shape (x, y, 3)

    Returns:
    a numpy array of shape (2x, 2y, 3)
    """
    # Extract the dimensions of each quadrant
    x, y = img_list[0].shape[0], img_list[0].shape[1]

    # Create an empty array to hold the merged image
    merged_img = np.zeros((2 * x, 2 * y, 3), dtype=np.uint8)

    # Combine the 4 quadrants into a single image
    merged_img[:x, :y, :] = img_list[0]
    merged_img[:x, y:, :] = img_list[1]
    merged_img[x:, :y, :] = img_list[2]
    merged_img[x:, y:, :] = img_list[3]

    return merged_img


TESTIMG = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
                    [[0, 0, 0], [128, 255, 128], [255, 255, 255]]], dtype="uint8")

grid = GridPlot((1, 2), (3, 1))
grid.AddImg(TESTIMG, 1, "testimg")
grid.AddImg(TESTIMG[:, :, 0].reshape(3, 3, 1), 2, "TESTIMG R")
grid.Show()

Save(TESTIMG, f"{BASEDIR}/tmp/TESTIMG.png")
Read(f"{BASEDIR}/tmp/TESTIMG.png")


# %% [markdown]
# ### 全色画像の生成

# %%
TILESIZE = 8
IMGSIZE = 4096
IMGSIZE**2 // TILESIZE**2

# %%
rgbstep**3

# %%
TILESIZE = 8
IMGSIZE = 4096
rgbstep = RGBStep(IMGSIZE, TILESIZE)
INDIR = f"{BASEDIR}/IN"

g = GridPlot((1, 4), (20, 10))

if TILESIZE == 1:
    c = list(itertools.product(
        np.arange(0, 256, 1),
        np.arange(0, 256, 1),
        np.arange(0, 256, 1)
    ))
    full_img = np.array(c, dtype='uint8').reshape(4096, 4096, 3)

else:
    rgb = np.linspace(0, 255, rgbstep, dtype=np.uint8)
    # BLOCKSIZE*BLOCKSIZE の単色画像を作成
    tiles = np.stack(np.meshgrid(rgb, rgb, rgb, indexing='ij'), axis=-1)
    tiles = tiles.reshape((-1, 1, 1, 3))

    # IMGSIZE*IMGSIZE の画像を作成
    tile_N = int(np.ceil(np.sqrt(len(tiles))))
    full_img = np.zeros((tile_N * TILESIZE, tile_N * TILESIZE, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        row = i // tile_N
        col = i % tile_N
        full_img[row * TILESIZE:(row + 1) * TILESIZE, col *
                 TILESIZE:(col + 1) * TILESIZE] = tile

    if full_img.shape[0] % 2 != 0 or full_img.shape[1] % 2 != 0:
        g.AddImg(full_img, 1, "before Padding")
        print(full_img.shape)
        full_img = Padding(full_img, targetX=IMGSIZE, targetY=IMGSIZE)

g.AddImg(full_img, 2, "result")
g.Show()
Save(full_img, f"{INDIR}/RGB_tile{TILESIZE}.png")

# %% [markdown]
# ### プリセット作成

# %%
TILESIZE = 8
IMGSIZE = 4096
# PRESETNAME="CU11"
PRESETNAME = "nostalgia"
# PRESETNAME="YM01"
OUTDIR = f"{BASEDIR}/OUT/tile{TILESIZE}"
INDIR = f"{BASEDIR}/IN"

rgbstep = RGBStep(IMGSIZE, TILESIZE)

g = GridPlot((1, 3))

OUT = Read(f"{OUTDIR}/{PRESETNAME}_tile{TILESIZE}.jpg")
g.AddImg(OUT, 2, f"OUT{OUT.shape[:2]}")

c = np.arange(TILESIZE // 2, IMGSIZE, TILESIZE)  # tileの中心ピクセル
OUT = OUT[np.ix_(c, c)]  # tileの中心以外を捨てる
g.AddImg(OUT, 3, f"OUTcenter{OUT.shape[:2]}")

IN = Read(f"{INDIR}/RGB_tile{TILESIZE}.png")
IN = IN[np.ix_(c, c)]
g.AddImg(IN, 1, f"INcenter{IN.shape[:2]}")
# Save(OUT,"nostalgia_OUT.png")
g.Show()


# LUT出力

I = IN.reshape((-1, 3))
order = np.lexsort((I[:, 0], I[:, 1], I[:, 2]))  # 3次元目、2次元目、1次元目の順でソート -> キーを取得
print(I)
O = OUT.reshape((-1, 3))
lut = (O[order] / 255)
print(lut)

directory = f"{BASEDIR}/filters/{rgbstep}"

if not os.path.exists(directory):
    os.makedirs(directory)

with open(f"{directory}/{PRESETNAME}.cube", 'w') as f:
    f.write(f"# Created by: babnomiyako\n")
    f.write(f"\n")
    f.write(f"# meta data\n")
    f.write(f"LUT_3D_SIZE {rgbstep}\n")
    f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
    f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
    f.write(f"\n")
    for row in lut:
        line = ' '.join(map(str, row))
        f.write(line + '\n')


# %%
# 0~255の整数を持つ(256, 256, 256, 3)のndarrayを、(256,256,256,3)に転置。3次元座標を表す
map_sparse = np.indices((256, 256, 256), dtype="uint8").transpose(1, 2, 3, 0)
# map_[IN[R],IN[G],IN[B]]=OUT [R,G,B]
map_sparse[IN[:, :, 0], IN[:, :, 1], IN[:, :, 2]] = OUT


# g=GridPlot((1,2))
# g.AddImg(IN,1,IN.shape)
# g.AddImg( map_[IN[:,:,0], IN[:,:,1], IN[:,:,2]],2)
# g.Show()

directory = f'{BASEDIR}/filters/{rgbstep}'
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(f'{directory}/{PRESETNAME}', map_sparse)

# 補完
if TILESIZE == 1:
    map_full = map_sparse
else:

    # Generate a set of coordinates for interpolation
    all_coords = np.meshgrid(np.arange(0, 256, 1),
                             np.arange(0, 256, 1),
                             np.arange(0, 256, 1),
                             indexing='ij')
    all_coords = np.stack(all_coords, axis=-1).astype(np.uint8)

    # Create the interpolator object
    c = np.linspace(0, 255, rgbstep, dtype="uint8")
    interpolator = RegularGridInterpolator((c, c, c), map_sparse[np.ix_(c, c, c)])

    # Interpolate the values at the new coordinates
    map_full = interpolator(all_coords).astype("uint8")

# %% [markdown]
# ### プリセット適用

# %%
TO = Read(rf'{BASEDIR}/kobako.png')
g = GridPlot((1, 4), (20, 10))
g.AddImg(TO, 1, "(Original)")

i = 2
for PRESETNAME in ["CU11", "nostalgia", "YM01"]:
    M = np.load(f'{BASEDIR}/filters/{PRESETNAME}.npy')
    TO2 = M[TO[:, :, 0], TO[:, :, 1], TO[:, :, 2]]
    g.AddImg(TO2, i, PRESETNAME)
    i += 1
g.Show()

# %%
shape = M.shape
r, g, b = np.indices(shape[:-1])
coords = np.stack([r, g, b], axis=-1)
DIFF = M - coords

plt.plot(
    DIFF[:, :, :, 0]
)

# %%
plt.plot(np.arange(0, 256, 1), M[:, 90, 50, 0])

# %%
# for PRESETNAME in ["CU11","nostalgia","YM01"]:
PRESETNAME = "nostalgia"
M = np.load(f'filters/{PRESETNAME}.npy')
M[:, :, :, 0]
plt.hist(M[:, :, :, 0].flatten(), alpha=0.4, bins=50)
plt.hist(M[:, :, :, 1].flatten(), alpha=0.4, bins=50)
plt.hist(M[:, :, :, 2].flatten(), alpha=0.4, bins=50)


# %% [markdown]
# ### 調節

# %%
# test colorsys
imgdat = np.array(
    [[[255, 0, 0], [0, 255, 0]],
     [[0, 0, 255], [100, 100, 50]]], dtype="uint8"
)
plt.imshow(Image.fromarray(imgdat))
plt.show()

a = np.array([[colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
             for r, g, b in i]for i in imgdat])

fig, ax = plt.subplots()
im = ax.imshow(a[:, :, 0], cmap='Greys', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)  # 凡例バーを表示
plt.show()


# %%
def my_func(a):
    """Average first and last element of a 1-D array"""
    return a.sum()


b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.apply_along_axis(my_func, 0, b)


# %%
import colorsys
in_[0, 1]


a = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in in_]

# %% [markdown]
# 色相Hは、RGBの最大値と最小値が同じ場合は0とする。そうでない場合は、
#
# 最大値がRのとき: (G - B) / (最大値 - 最小値) * 60
# 最大値がGのとき: (B - R) / (最大値 - 最小値) * 60 + 120
# 最大値がBのとき: (R - G) / (最大値 - 最小値) * 60 + 240
# 彩度Sは、RGBの最大値と最小値が同じ場合は0とする。そうでない場合は、
#
# 明度Vが0.5以下のとき: (最大値 - 最小値) / (最大値 + 最小値)
# 明度Vが0.5より大きいとき: (最大値 - 最小値) / (2 - 最大値 - 最小値)
# 明度Vは、RGBの最大値をそのまま用いる。
#
# 輝度Lは、RGBの平均をそのまま用いる。
