#
# import numpy as np
#
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
#
# import tkinter as tk
# from PIL import Image
#
# from typing import Union
#
# from autofit.src.composite_function import CompositeFunction
# from autofit.src.package import pkg_path, logger
#
# class ImageHandler :
#
#     def __init__(self):
#
#         self.image_r: float = 1
#         self.image_path: str = ""
#         self.image: Union[None,tk.PhotoImage] = None
#
#         self.background_labels = ["Default","White","Dark","Black"]
#         self.dataaxis_labels   = ["Default","White"]
#         self.fit_colour_labels = ["Default","White","Black"]
#
#         self.default_bg_colour: str = "Default"
#         self.default_dataaxes_colour: str = "Default"
#         self.default_fit_colour: str = "Default"
#
#         self.bg_color: tuple[float, float, float] = (112 / 255, 146 / 255, 190 / 255)
#         self.fit_color: tuple[float, float, float] = (1., 0., 0.)
#         self.dataaxes_color: tuple[float, float, float] = (1., 1., 1.)
#
#         self._show_error_bands = 0
#
#     def load_splash_image(self, default_width, os_height):
#         self.image_path = f"{pkg_path()}/splash.png"
#
#         img_raw = Image.open(self.image_path)
#         if default_width < 2 and self.image_r == 1 :
#             self.image_r = (8/9) * os_height / (2*img_raw.height)
#         img_resized = img_raw.resize((round(img_raw.width * self.image_r),
#                                       round(img_raw.height * self.image_r)))
#         self.image_path = f"{self.image_path[:-4]}_mod.png"
#         img_resized.save(fp=self.image_path)
#
#         self.image = tk.PhotoImage(file=self.image_path)
#
#     def set_up_axes(self, axes):
#         axes.tick_params(color=self.dataaxes_color, labelcolor=self.dataaxes_color)
#         axes.xaxis.label.set_color(self.dataaxes_color)
#         axes.yaxis.label.set_color(self.dataaxes_color)
#         for spine in axes.spines.values():
#             spine.set_edgecolor(self.dataaxes_color)
#         if axes.get_xlim()[0] > 0:
#             axes.set_xlim([0, axes.get_xlim()[1]])
#         elif axes.get_xlim()[1] < 0:
#             axes.set_xlim([axes.get_xlim()[0], 0])
#         if axes.get_ylim()[0] > 0:
#             axes.set_ylim([0, axes.get_ylim()[1]])
#         elif axes.get_ylim()[1] < 0:
#             axes.set_ylim([axes.get_ylim()[0], 0])
#
#     @staticmethod
#     def apply_logx_axes(axes, xmin, xmax):
#         logger("Setting log x-scale in show_data")
#         log_min, log_max = np.log(xmin), np.log(xmax)
#         logger(log_min, log_max, np.exp(log_min), np.exp(log_max))
#         axes.set_xlim(
#             [np.exp(log_min - (log_max - log_min) / 10), np.exp(log_max + (log_max - log_min) / 10)])
#         axes.set(xscale="log")
#         axes.spines['right'].set_visible(False)
#
#     @staticmethod
#     def apply_linx_axes(axes, xmin, xmax):
#
#         axes.set(xscale="linear")
#         axes.spines['left'].set_position(('data', 0.))
#         axes.spines['right'].set_position(('data', 0.))
#
#         log_deltaX = np.log10(xmin - xmax if xmax > xmin else 10) // 1
#         if log_deltaX > 4:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2E}"))
#         elif 0 <= log_deltaX <= 4:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(
#                 lambda x, pos: "" if x == 0 else (f"{x:.1F}" if (x - np.trunc(x)) ** 2 > 1e-10 else f"{int(x)}")))
#
#         elif log_deltaX == -1:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2F}"))
#         elif log_deltaX == -2:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.3F}"))
#         elif log_deltaX == -3:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.4F}"))
#         elif log_deltaX == -4:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.5F}"))
#         else:
#             axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2E}"))
#     @staticmethod
#     def apply_logy_axes(axes, ymin, ymax):
#         logger("Setting log y-scale in show_data")
#         axes.set(yscale="log")
#         log_min, log_max = np.log(ymin), np.log(ymax)
#         axes.set_ylim(
#             [np.exp(log_min - (log_max - log_min) / 10), np.exp(log_max + (log_max - log_min) / 10)])
#         axes.spines['top'].set_visible(False)
#     @staticmethod
#     def apply_liny_axes(axes, ymin, ymax):
#
#         axes.set(yscale="linear")
#         axes.spines['top'].set_position(('data', 0.))
#         axes.spines['bottom'].set_position(('data', 0.))
#         # axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.1F}"))
#
#         log_deltaY = np.log10(ymax - ymin if ymax > ymin else 10) // 1
#         if log_deltaY > 4:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2E}"))
#         elif 0 <= log_deltaY <= 4:
#             axes.yaxis.set_major_formatter(
#                 ticker.FuncFormatter(
#                     lambda x, pos: "" if x == 0 else (f"{x:.1F}" if (x - np.trunc(x)) ** 2 > 1e-10 else f"{int(x)}")
#                 )
#             )
#
#         elif log_deltaY == -1:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2F}"))
#         elif log_deltaY == -2:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.3F}"))
#         elif log_deltaY == -3:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.4F}"))
#         elif log_deltaY == -4:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.5F}"))
#         else:
#             axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "" if x == 0 else f"{x:.2E}"))
#
#     def fix_remaining(self, axes, xmin, xmax, ymin, ymax, xlabel):
#         axes.set_facecolor(self.bg_color)
#
#         #  proportion between xmin and xmax where the zero lies
#         # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
#         tx = max(0., -xmin / (xmax - xmin))
#         ty = max(0., -ymin / (max(ymax - ymin, 1e-5)))
#         offset_X, offset_Y = -0.1, 0.0  # how much of the screen is taken by the x and y spines
#
#         if ymin < 0 :
#             axes.xaxis.set_label_coords(1.050-0.01*len(xlabel), offset_Y + ty-0.05)
#         else :
#             axes.xaxis.set_label_coords(0.5, offset_Y + ty-0.03)
#             plt.tight_layout()
#
#         axes.yaxis.set_label_coords(offset_X + tx, +0.750)
#
#         plt.savefig(self.image_path, facecolor=self.bg_color)
#         plt.show()
#
#     @staticmethod
#     def show():
#         plt.show()
#
#     def show_data(self, data_handler):
#         self.image_path = f"{pkg_path()}/plots/front_end_current_plot.png"
#         # create a scatter plot of the first file
#
#         x_points = data_handler.unlogged_x_data
#         y_points = data_handler.unlogged_y_data
#         sigma_x_points = data_handler.unlogged_sigmax_data
#         sigma_y_points = data_handler.unlogged_sigmay_data
#
#         print(x_points)
#         print(y_points)
#         min_X, max_X = min(x_points), max(x_points)
#         min_Y, max_Y = min(y_points), max(y_points)
#
#         plt.close()
#         plt.figure(facecolor=self.bg_color,
#                    figsize=(6.4*self.image_r,4.8*self.image_r),
#                    dpi=100+int( np.log10( len(x_points) ) )
#                   )
#
#         plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o',
#                      color=self.dataaxes_color)
#         plt.show()
#         plt.xlabel(data_handler.x_label)
#         plt.ylabel(data_handler.y_label)
#         axes = plt.gca()
#         self.set_up_axes(axes)
#
#         if data_handler.logx_flag:
#             self.apply_logx_axes(axes,min_X, max_X)
#         else:
#             self.apply_linx_axes(axes,min_X, max_X)
#
#         if data_handler.logy_flag:
#             self.apply_logy_axes(axes, min_Y, max_Y)
#         else:
#             self.apply_liny_axes(axes, min_Y, max_Y)
#
#         self.fix_remaining(axes, min_X, max_X, min_Y, max_Y, data_handler.x_label)
#
#     @staticmethod
#     def y_uncertainty(xval, data_handler, model : CompositeFunction, cov):
#         # simple propagation of uncertainty with first-order finite differnece approximation of parameter derivatives
#         par_derivs = []
#         for idx, arg in enumerate(model.args[:]):
#             shifted_args = model.args.copy()
#             shifted_args[idx] = arg + abs(arg) / 1e5
#             shifted_model = model.copy()
#             shifted_model.args = shifted_args
#             if data_handler.logx_flag and data_handler.logy_flag:
#                 par_derivs.append((shifted_model.eval_at(xval, X0=data_handler.X0, Y0=data_handler.Y0)
#                                    - model.eval_at(xval, X0=data_handler.X0, Y0=data_handler.Y0))
#                                   / (abs(arg) / 1e5))
#             elif data_handler.logx_flag:
#                 par_derivs.append((shifted_model.eval_at(xval, X0=data_handler.X0)
#                                    - model.eval_at(xval, X0=data_handler.X0)) / (abs(arg) / 1e5))
#             elif data_handler.logy_flag:
#                 par_derivs.append((shifted_model.eval_at(xval, Y0=data_handler.Y0)
#                                    - model.eval_at(xval, Y0=data_handler.Y0)) / (abs(arg) / 1e5))
#             else:
#                 par_derivs.append((shifted_model.eval_at(xval)
#                                    - model.eval_at(xval)) / (abs(arg) / 1e5))
#
#         partial_V_partial = 0
#         for i, _ in enumerate(model.args):
#             for j, _ in enumerate(model.args):
#                 partial_V_partial += par_derivs[i] * cov[i, j] * par_derivs[j]
#         return np.sqrt(partial_V_partial)
#
#     def save_show_fit_image(self, data_handler, current_model, current_cov):
#
#         plot_model = current_model.copy()
#
#         x_points = data_handler.unlogged_x_data
#         y_points = data_handler.unlogged_y_data
#         sigma_x_points = data_handler.unlogged_sigmax_data
#         sigma_y_points = data_handler.unlogged_sigmay_data
#
#         min_X, max_X = min(x_points), max(x_points)
#         min_Y, max_Y = min(y_points), max(y_points)
#
#         smooth_x_for_fit = np.linspace(min_X, max_X, 4 * len(x_points))
#
#         if data_handler.logx_flag and data_handler.logy_flag:
#             fit_vals = [plot_model.eval_at(xi, X0=data_handler.X0, Y0=data_handler.Y0)
#                         for xi in smooth_x_for_fit]
#         elif data_handler.logx_flag:
#             fit_vals = [plot_model.eval_at(xi, X0=data_handler.X0) for xi in smooth_x_for_fit]
#         elif data_handler.logy_flag:
#             fit_vals = [plot_model.eval_at(xi, Y0=data_handler.Y0) for xi in smooth_x_for_fit]
#         else:
#             fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]
#
#         plt.close()
#         plt.figure(facecolor=self.bg_color,
#                    figsize=(6.4*self.image_r,4.8*self.image_r),
#                    dpi=100+int( np.log10( len(x_points) ) )
#                   )
#         plt.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color=self.dataaxes_color)
#
#         plt.plot(smooth_x_for_fit, fit_vals, '-', color=self.fit_color)
#         if self._show_error_bands in [1, 3]:
#             unc_list = [self.y_uncertainty(xi, data_handler, current_model, current_cov) for xi in smooth_x_for_fit]
#             upper_error_vals = [val + unc for val, unc in zip(fit_vals, unc_list)]
#             lower_error_vals = [val - unc for val, unc in zip(fit_vals, unc_list)]
#
#             plt.fill_between(smooth_x_for_fit,lower_error_vals,upper_error_vals,color=self.fit_color,alpha=0.5)
#         if self._show_error_bands in [2, 3]:
#             unc_list = [self.y_uncertainty(xi, data_handler, current_model, current_cov) for xi in smooth_x_for_fit]
#             upper_2error_vals = [val + 2 * unc for val, unc in zip(fit_vals, unc_list)]
#             lower_2error_vals = [val - 2 * unc for val, unc in zip(fit_vals, unc_list)]
#
#             plt.fill_between(smooth_x_for_fit,lower_2error_vals,upper_2error_vals,color=self.fit_color,alpha=0.5)
#
#
#         plt.xlabel(data_handler.x_label)
#         plt.ylabel(data_handler.y_label)
#         axes: plt.axes = plt.gca()
#         self.set_up_axes(axes)
#
#         if data_handler.logx_flag:
#             self.apply_logx_axes(axes, min_X, max_X)
#         else:
#             self.apply_linx_axes(axes, min_X, max_X)
#         if data_handler.logy_flag:
#             self.apply_logy_axes(axes, min_Y, max_Y)
#         else:
#             self.apply_liny_axes(axes, min_Y, max_Y)
#         axes.set_facecolor(self.bg_color)
#
#
#         #  tx is the proportion between xmin and xmax where the zero lies
#         # x(tx) = xmin + (xmax - xmin)*tx with 0<tx<1 so
#         tx = max(0., -min_X / (max_X - min_X))
#         ty = max(0., -min_Y / (max_Y - min_Y))
#         offset_X, offset_Y = -0.1, 0.0  # how much of the screen is taken by the x and y spines
#
#         if min_Y < 0 or min(fit_vals) < 0:
#             axes.xaxis.set_label_coords(1.050-0.015*len(data_handler.x_label), offset_Y + ty-0.05)
#         else :
#             axes.xaxis.set_label_coords(0.5, offset_Y + ty-0.03)
#         axes.yaxis.set_label_coords(offset_X + tx, +0.750)
#
#         plt.tight_layout()
#         plt.savefig(self.image_path, facecolor=self.bg_color)
#
#     def save_show_fit_all(self, current_handler, data_handlers, current_model, args_list, current_cov):
#
#         plot_model = current_model.copy()
#
#         num_sets = len(data_handlers)
#         abs_minX, abs_minY =  1e5,  1e5
#         abs_maxX, abs_maxY = -1e5, -1e5
#
#         sum_len = 0
#
#         plt.close()
#         plt.figure(facecolor=self.bg_color,
#                    figsize=(6.4*self.image_r,4.8*self.image_r),
#                    dpi=100+int( np.log10( len(current_handler.unlogged_x_data) ) )
#                   )
#         axes: plt.axes = plt.gca()
#
#         for idx, (handler, args) in enumerate(zip(data_handlers, args_list)):
#
#             x_points = handler.unlogged_x_data
#             y_points = handler.unlogged_y_data
#             sigma_x_points = handler.unlogged_sigmax_data
#             sigma_y_points = handler.unlogged_sigmay_data
#
#             sum_len += len(x_points)
#             smooth_x_for_fit = np.linspace(x_points[0], x_points[-1], 4 * len(x_points))
#             plot_model.args = args
#             logger(f"{plot_model.args=}")
#             if handler.logx_flag and handler.logy_flag:
#                 fit_vals = [plot_model.eval_at(xi, X0=handler.X0, Y0=handler.Y0)
#                             for xi in smooth_x_for_fit]
#             elif handler.logx_flag:
#                 fit_vals = [plot_model.eval_at(xi, X0=handler.X0) for xi in smooth_x_for_fit]
#             elif handler.logy_flag:
#                 fit_vals = [plot_model.eval_at(xi, Y0=handler.Y0) for xi in smooth_x_for_fit]
#             else:
#                 fit_vals = [plot_model.eval_at(xi) for xi in smooth_x_for_fit]
#
#             col_tuple = [(icol / max(self.dataaxes_color) if max(self.dataaxes_color) > 0 else 1)
#                          * (idx / num_sets) for icol in self.dataaxes_color]
#
#             axes.errorbar(x_points, y_points, xerr=sigma_x_points, yerr=sigma_y_points, fmt='o', color=col_tuple)
#             plt.plot(smooth_x_for_fit, fit_vals, '-', color=col_tuple)
#
#             min_X, max_X = min(x_points), max(x_points)
#             min_Y, max_Y = min(y_points), max(y_points)
#
#             if min_X < abs_minX:
#                 abs_minX = min_X
#             if min_Y < abs_minY:
#                 abs_minY = min_Y
#             if max_X > abs_maxX:
#                 abs_maxX = max_X
#             if max_Y > abs_maxY:
#                 abs_maxY = max_Y
#
#             plt.draw()
#
#         # also add average fit
#         smooth_x_for_fit = np.linspace(abs_minX, abs_maxX, sum_len)
#         if current_handler.logx_flag and current_handler.logy_flag:
#             fit_vals = [plot_model.eval_at(xi, X0=current_handler.X0, Y0=current_handler.Y0)
#                         for xi in smooth_x_for_fit]
#         elif current_handler.logx_flag:
#             fit_vals = [plot_model.eval_at(xi, X0=current_handler.X0) for xi in smooth_x_for_fit]
#         elif current_handler.logy_flag:
#             fit_vals = [plot_model.shown_model.eval_at(xi, Y0=current_handler.Y0) for xi in smooth_x_for_fit]
#         else:
#             fit_vals = [plot_model.shown_model.eval_at(xi) for xi in smooth_x_for_fit]
#
#         plt.plot(smooth_x_for_fit, fit_vals, '-', color=self.fit_color)
#         if self._show_error_bands in [1, 3]:
#             unc_list = [self.y_uncertainty(xi, current_handler, current_model, current_cov) for xi in smooth_x_for_fit]
#             upper_error_vals = [val + unc for val, unc in zip(fit_vals, unc_list)]
#             lower_error_vals = [val - unc for val, unc in zip(fit_vals, unc_list)]
#
#             plt.plot(smooth_x_for_fit, upper_error_vals, '--', color=self.fit_color)
#             plt.plot(smooth_x_for_fit, lower_error_vals, '--', color=self.fit_color)
#         if self._show_error_bands in [2, 3]:
#             unc_list = [self.y_uncertainty(xi, current_handler, current_model, current_cov) for xi in smooth_x_for_fit]
#             upper_2error_vals = [val + 2 * unc for val, unc in zip(fit_vals, unc_list)]
#             lower_2error_vals = [val - 2 * unc for val, unc in zip(fit_vals, unc_list)]
#
#             plt.plot(smooth_x_for_fit, upper_2error_vals, ':', color=self.fit_color)
#             plt.plot(smooth_x_for_fit, lower_2error_vals, ':', color=self.fit_color)
#
#         self.set_up_axes(axes)
#
#         if current_handler.logx_flag:
#             self.apply_logx_axes(axes, abs_minX, abs_maxX)
#         else:
#             self.apply_linx_axes(axes, abs_minX, abs_maxX)
#         if current_handler.logy_flag:
#             self.apply_logy_axes(axes, abs_minX, abs_minY)
#         else:
#             self.apply_liny_axes(axes, abs_minX, abs_minY)
#
#         self.fix_remaining(axes, abs_minX, abs_maxX, abs_minY, abs_maxX, current_handler.x_label)
#
#     def switch_image(self):
#         self.image = tk.PhotoImage(file=self.image_path)
