import io
import base64
import matplotlib.pyplot as plt

def plt_to_base64():
    """Converts matplotlib plot to a base64 string compatible with `vizstack.Image`."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()