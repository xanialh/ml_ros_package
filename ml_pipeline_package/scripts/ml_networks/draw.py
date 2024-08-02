import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_fcn_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define positions and sizes
    encoder_pos = [(1, 8), (1, 6), (1, 4), (1, 2), (1, 0)]
    decoder_pos = [(8, 8), (8, 6), (8, 4), (8, 2), (8, 0)]

    # Draw Input Layer
    ax.add_patch(patches.Rectangle((0, 4), 1, 2, edgecolor='black', facecolor='lightgray', lw=2))
    ax.text(0.5, 5, 'Input Image\n(1 channel)', fontsize=10, ha='center')

    # Draw Encoder Layers
    encoder_layers = [
        ('Conv2d\n(1x64, 3x3)', 'blue', (2, 1)),
        ('ReLU', 'blue', (2, 1)),
        ('Conv2d\n(64x64, 3x3)', 'blue', (2, 1)),
        ('MaxPool2d\n(2x2)', 'blue', (2, 1)),
        ('Conv2d\n(64x128, 3x3)', 'blue', (2, 1)),
        ('ReLU', 'blue', (2, 1)),
        ('Conv2d\n(128x128, 3x3)', 'blue', (2, 1)),
        ('MaxPool2d\n(2x2)', 'blue', (2, 1))
    ]
    
    for (label, color, size), pos in zip(encoder_layers, encoder_pos):
        ax.add_patch(patches.Rectangle((pos[0], pos[1]), size[0], size[1], edgecolor=color, facecolor=f'{color}', lw=2))
        ax.text(pos[0] + size[0]/2, pos[1] + size[1]/2, label, fontsize=10, ha='center')

    # Draw Decoder Layers
    decoder_layers = [
        ('ConvTranspose2d\n(128x128, 4x4)', 'red', (2, 1)),
        ('ReLU', 'red', (2, 1)),
        ('Conv2d\n(128x64, 3x3)', 'red', (2, 1)),
        ('Conv2d\n(64x32, 3x3)', 'red', (2, 1)),
        ('ConvTranspose2d\n(32x3, 4x4)', 'red', (2, 1))
    ]
    
    for (label, color, size), pos in zip(decoder_layers, decoder_pos):
        ax.add_patch(patches.Rectangle((pos[0], pos[1]), size[0], size[1], edgecolor=color, facecolor=f'{color}', lw=2))
        ax.text(pos[0] + size[0]/2, pos[1] + size[1]/2, label, fontsize=10, ha='center')

    # Draw Output Layer
    ax.add_patch(patches.Rectangle((16, 4), 1, 2, edgecolor='black', facecolor='lightgray', lw=2))
    ax.text(16.5, 5, 'Output Image\n(3 channels)', fontsize=10, ha='center')

    # Function to Draw Arrow
    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Connect Input to Encoder
    draw_arrow((0.5, 5), (encoder_pos[0][0], encoder_pos[0][1] + 0.5))

    # Connect Encoder to Decoder
    draw_arrow((encoder_pos[-1][0] + 2, encoder_pos[-1][1] + 0.5), (decoder_pos[0][0], decoder_pos[0][1] + 0.5))

    # Connect Decoder Layers
    for i in range(len(decoder_pos) - 1):
        draw_arrow((decoder_pos[i][0] + 2, decoder_pos[i][1] + 0.5), (decoder_pos[i + 1][0], decoder_pos[i + 1][1] + 0.5))

    # Connect Decoder to Output Layer
    draw_arrow((decoder_pos[-1][0] + 2, decoder_pos[-1][1] + 0.5), (16, 5))

    # Show the Plot
    plt.show()

# Call the function to draw the diagram
draw_fcn_diagram()
