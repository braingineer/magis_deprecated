import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import colorsys

def enforce_arguments(out_func):
    def func(*args):
        if isinstance(args[0], (list, tuple)) and len(args) == 1:
            args = args[0]
        elif isinstance(args[0], (list,tuple)):
            raise Exception("Use map to apply to multiple colors")
        args = tuple(args)
        return out_func(*args)
    return func


class Convert(object):
    @enforce_arguments
    def normalized_adjusted_hsv(self, *hsv):
        h,s,v = hsv
        if h < 0:
            h += 1.0
        return colorsys.hsv_to_rgb(h,s,v)

    @enforce_arguments
    def normalized_hsv(self, *hsv):
        return colorsys.hsv_to_rgb(*hsv)

    @enforce_arguments
    def scaled_hsv(self, *hsv):
        h,s,v = hsv
        h /= 360.
        s /= 100.
        v /= 100.
        return self.normalized_hsv(h,s,v)

Convert = Convert()

class Plot(object):
    def __init__(self):
        self._converters = dict(filter(lambda _: _[0]!="_", Convert.__dict__.keys()))
        self._current_converter = "normalized_hsv"

    @property
    def converter(self):
        return getattr(Convert, self._current_converter)

    @converter.setter
    def converter(self, new_converter):
        if new_converter not in self._converters.keys():
            raise ValueError("{} not a valid converter;" +
                             " Use one of: {}".format(new_converter, 
                                                      ", ".join(self._converters.keys()))) 

    def plot(self, colors, predictions, show=False, prediction_kwargs=None):
        
        if not isinstance(colors[0], (list, tuple)):
            colors = [colors]
            predictions = [predictions]
        fig, axes = plt.subplots(len(colors), 2)

        prediction_kwargs = prediction_kwargs or {}

        axes = axes.reshape(len(colors), 2)
        for i, (color, prediction) in enumerate(zip(colors, predictions)):
            color = self.converter(*color)
            Plot._color(color, axes[i, 0])
            Plot._predictions(prediction, axes[i, 1], **prediction_kwargs)
        return fig

    def _color(self, color, ax):
        r = mpl.patches.Rectangle((0,0), 100,100, color=color)
        ax.add_patch(r)
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.axis('off')

    def _predictions(self, predictions, ax, 
                           title_text="Predictions", title_font=15, word_font=10, 
                           sepline_y=0.88, sepline_x0=0.05, sepline_x1=0.7, 
                           word_spacing=0.07, word_colsize=20, number_colsize=5, 
                           left_x=0.05, title_y=0.9):
        '''
        Plot a series of text predictions, possibly with probability values.
        Plots a title to the predictions as well.

        Args: 
            predictions:    list of words, or
                            list of 2-tuples (word, number), or
                            dict of {word: number}
            ax:             ax to plot on
            title_text:     text to display (inside the ax)
            title_font:     font size
            word_font:      font size
            sepline_y:      a horizontal line to separate title's y position
            sepline_x0:     line's starting x
            sepline_x1:     line's ending x
            word_spacing:   y-distance between each word
            word_colsize:   in format string, number of characters to left justify with
            number_colsize: in format string, number of characters to right justify with
            left_x:         left edge of the title and words
            title_y:        y position of the title 
        '''
        ax.text(left_x, title_y, "Predictions", fontsize=title_font, family='monospace')
        ax.plot([sepline_x0, sepline_x1], [sepline_y, sepline_y], color='black')
        ax.axis('off')
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        
        starting = sepline_y - word_spacing
        # using this formatting to make a formatting str.. even though I hate the % format
        format_str = "{:<%d}{:>%d.2f}" % (word_colsize, number_colsize)

        if isinstance(predictions, dict): 
            predictions = predictions.items()

        for i, p in enumerate(predictions):
            if isinstance(p, (list, tuple)) and len(p) == 2:
                p, number = p
                p = format_str.format(p, float(number))

            ax.text(left_x, starting-i*word_spacing, p, fontsize=word_font, family='monospace')


Plot = Plot()