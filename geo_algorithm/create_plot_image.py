from PIL import Image
import os
import math

# 画像が保存されているディレクトリ
input_dir = 'community_analysis_maps/survey/幅員4m以上の道路に面していない住宅の割合'
# 出力する画像ファイルのディレクトリ
output_dir = 'result/combined_images_幅員4m以上の道路に面していない住宅の割合'
os.makedirs(output_dir, exist_ok=True)

# 画像ファイルのみを取得
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
sample_image_path = os.path.join(input_dir, file_list[0])
sample_image = Image.open(sample_image_path)
image_width, image_height = sample_image.size

# グリッドのサイズ
columns = 2
rows = 3
images_per_grid = columns * rows

# 画像をグリッドに配置
for i in range(0, len(file_list), images_per_grid):
    # 各グリッドの最大幅と高さを計算
    max_widths = [0] * columns
    max_heights = [0] * rows

    for j in range(images_per_grid):
        index = i + j
        if index >= len(file_list):
            break
        filename = file_list[index]
        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size

            col = j % columns
            row = j // columns
            max_widths[col] = max(max_widths[col], img_width)
            max_heights[row] = max(max_heights[row], img_height)

        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    # キャンバスのサイズを計算
    total_width = sum(max_widths)
    total_height = sum(max_heights)

    # 各グリッドのキャンバスを作成（背景色を白に設定）
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    current_x, current_y = 0, 0

    for j in range(images_per_grid):
        index = i + j
        if index >= len(file_list):
            break
        filename = file_list[index]
        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size

            if j % columns == 0 and j != 0:
                # 次の行に移動
                current_x = 0
                current_y += max_heights[j // columns - 1]

            combined_image.paste(img, (current_x, current_y))
            current_x += max_widths[j % columns]

        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    # 画像を保存
    output_file = os.path.join(output_dir, f'combined_image_{i // images_per_grid + 1}.png')
    combined_image.save(output_file)