import os
import re
totalhtml = ""
for i, filename in enumerate(os.listdir("result")):
    file_path = os.path.join("result", filename)
    if os.path.isfile(file_path) and filename.endswith(".html"):
        with open(file_path, "r") as f:
            html_content = f.read()
            picname1 = filename.replace(".html", "_ocr_boxes.jpg")
            picname2 = filename.replace(".html", "_table_boxes.jpg")
            # 使用正则删除掉所有<img>标签
            html_content = re.sub(r'<img[^>]*>', '', html_content)
            bodypos = html_content.find("</body>")
            if bodypos != -1:
                html_content = html_content[:bodypos] + \
                    f'<img src="{picname1}" style="max-width:100%;height:auto;"/><img src="{picname2}" style="max-width:100%;height:auto;"/>' + \
                    html_content[bodypos:]
            else:
                html_content += f'<img src="{picname1}" style="max-width:100%;height:auto;"/><img src="{picname2}" style="max-width:100%;height:auto;"/>'
        with open(file_path, "w") as f:
            f.write(html_content)
        # 添加到总html中
        totalhtml += f"<h1>case {int(i/3)}</h1>" + html_content

# 将总html写入文件
with open(os.path.join("result", "total.html"), "w") as f:
    f.write(totalhtml)

