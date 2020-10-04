import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_input_image(x, opts):
	if opts.label_nc == 0:
		return tensor2im(x)
	elif opts.label_nc == 1:
		return tensor2sketch(x)
	else:
		return tensor2map(x)


def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def tensor2map(var):
	mask = np.argmax(var.data.cpu().numpy(), axis=0)
	colors = get_colors()
	mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
	for class_idx in np.unique(mask):
		mask_image[mask == class_idx] = colors[class_idx]
	mask_image = mask_image.astype('uint8')
	return Image.fromarray(mask_image)


def tensor2sketch(var):
	im = var[0].cpu().detach().numpy()
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	im = (im * 255).astype(np.uint8)
	return Image.fromarray(im)


# Visualization utils
def get_colors():
	# currently support up to 19 classes (for the celebs-hq-mask dataset)
	colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
			  [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
			  [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
	return colors


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')
