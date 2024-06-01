
def load_fonts(font_path):
    
    import os
    import matplotlib.font_manager as fm
    import matplotlib.ticker as plticker
    for x in os.listdir(font_path):
        if x != ".DS_Store":
            for y in os.listdir(f"{font_path}/{x}"):
                if y.split(".")[-1] == "ttf":
                    fm.fontManager.addfont(f"{font_path}/{x}/{y}")
                    try:
                        fm.FontProperties(weight=y.split("-")[-1].split(".")[0].lower(), fname=y.split("-")[0])
                    except Exception:
                        continue
                    
def load_mpl_style(style_path):
    
    import matplotlib.pyplot as plt
    plt.style.use(style_path)