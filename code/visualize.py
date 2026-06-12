import matplotlib.pyplot as plt

def vis(imgs, save_path, titles=None):
    """
        imgs: tensor or numpy, [r, c, h, w, rgb]
    """
    nrow = len(imgs)
    assert(nrow > 0)
    ncol = len(imgs[0])
    assert(ncol > 0)
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    plt.clf()
    DPI = 100
    
    fig_h = img_h * nrow / DPI
    fig_w = img_w * ncol / DPI
    fig = plt.figure(figsize=(fig_w, fig_h))
    # plt.title(self.title + ' (' + self.description + ')')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.tight_layout()

    for i in range(nrow * ncol):
        ax = plt.subplot(nrow, ncol, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None and i % ncol == 0:
            ax.set_title(titles[i // ncol])
        plt.imshow(imgs[i // ncol][i % ncol])
    plt.savefig(save_path)
    plt.close('all')
    return fig