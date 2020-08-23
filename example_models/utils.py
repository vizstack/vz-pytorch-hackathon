import io
import base64

from matplotlib import pyplot as plt


def plt_to_bytes():
    pic_iobytes = io.BytesIO()
    plt.savefig(pic_iobytes, format='png')
    pic_iobytes.seek(0)
    return base64.b64encode(pic_iobytes.read()).decode()