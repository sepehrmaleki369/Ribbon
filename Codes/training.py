import time
import numpy as np
import os
import logging
import torch
from . import utils
from skimage.morphology import skeletonize
from .scores import correctness_completeness_quality
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self,training_step, validation=None, valid_every=None,
                 print_every=None, save_every=None, save_path=None, save_objects={},
                 save_callback=None):

        self.training_step = training_step
        self.validation = validation
        self.valid_every = valid_every
        self.print_every = print_every
        self.save_every = save_every
        self.save_path = save_path
        self.save_objects = save_objects
        self.save_callback = save_callback

        self.starting_iteration = 0

        if self.save_path is None or self.save_path in ["", ".", "./"]:
            self.save_path = os.getcwd()

        self.results = {"training":{"epochs":[], "results":[]},
                        "validation": {"epochs":[], "results":[]}}

    def save_state(self, iteration):

        # Save to local experiments folder
        for name, object in self.save_objects.items():
            utils.torch_save(os.path.join(self.save_path, "{}_final.pickle".format(name)),
                             object.state_dict())

        utils.pickle_write(os.path.join(self.save_path, "results_final.pickle"),
                           self.results)
        
        # Also save to Google Drive with iteration number
        drive_checkpoint_path = "/content/drive/MyDrive/ribbs/october/RibbonSertac"
        utils.mkdir(drive_checkpoint_path)
        for name, object in self.save_objects.items():
            checkpoint_file = os.path.join(drive_checkpoint_path, f"checkpoint_epoch_{iteration}.pth")
            utils.torch_save(checkpoint_file, object.state_dict())
            logger.info(f"Saved checkpoint to {checkpoint_file}")

        if self.save_callback is not None:
            self.save_callback(iteration)

    def run_for(self, iterations):

        start_time = time.time()
        block_iter_start_time = time.time()

        for iteration in range(self.starting_iteration, self.starting_iteration+iterations+1):

            train_step_results = self.training_step(iteration)
            self.results['training']["results"].append(train_step_results)
            self.results['training']["epochs"].append(iteration)

            if self.print_every is not None:
                if iteration%self.print_every==0:
                    elapsed_time = (time.time() - start_time)//60
                    block_iter_elapsed_time = time.time() - block_iter_start_time

                    loss_v1 = train_step_results["loss"] if "loss" in train_step_results.keys() else None
                    loss_v2 = train_step_results["loss_2"] if "loss_2" in train_step_results.keys() else 0
                    to_print = "[{:0.0f}min][{:0.2f}s] - Epoch: {} (Train-batch Loss: {:0.6f}, {:0.6f})"
                    to_print = to_print.format(elapsed_time, block_iter_elapsed_time, iteration, loss_v1, loss_v2)

                    logger.info(to_print)
                    block_iter_start_time = time.time()

            if self.validation is not None and self.valid_every is not None:
                if iteration%self.valid_every==0 and iteration!=self.starting_iteration:
                    logger.info("validation...")
                    start_valid = time.time()

                    validation_results = self.validation(iteration)

                    self.results['validation']["results"].append(validation_results)
                    self.results['validation']["epochs"].append(iteration)
                    logger.info("Validation time: {:.2f}s".format(time.time()-start_valid))

            if self.save_every is not None:
                if iteration%self.save_every==0 and iteration!=self.starting_iteration:
                    start_saving = time.time()
                    self.save_state(iteration)
                    logger.info("Saving time: {:.2f}s".format(time.time()-start_saving))

class TrainingEpoch(object):

    def __init__(self, dataloader, ours=False, ours_start=0):

        self.dataloader = dataloader
        self.ours = ours
        self.ours_start = ours_start

    def __call__(self, iterations, network, optimizer, lr_scheduler, base_loss, our_loss):
        
        # Log which loss is being used every 10 epochs
        if iterations % 10 == 0:
            if self.ours and iterations >= self.ours_start:
                logger.info(f"Epoch {iterations}: Using SnakeSimpleLoss (adjusting annotations)")
            else:
                logger.info(f"Epoch {iterations}: Using MSELoss (baseline)")
        
        mean_loss = 0
        snake_adjusted_dmap_initial = None  # Store initial snake for visualization
        snake_adjusted_dmap = None  # Store final snake for visualization
        vis_images = None
        vis_labels = None
        vis_preds = None
        
        for batch_idx, (images, labels, graphs, slices, original_shapes) in enumerate(self.dataloader):

            images = images.cuda()
            labels = labels.cuda()
            
            preds = network(images.contiguous())
            
            if self.ours and iterations >= self.ours_start:
            # calls forward on loss here, and snake is adjusted
                loss = our_loss(preds, graphs, slices, None, original_shapes)
                # Store snake distance maps AND corresponding images for visualization (from first batch)
                if batch_idx == 0:
                    snake_adjusted_dmap_initial = getattr(our_loss, 'snake_dm_initial', None)
                    snake_adjusted_dmap = getattr(our_loss, 'snake_dm', None)
                    vis_images = images
                    vis_labels = labels
                    vis_preds = preds
            else:
                loss = base_loss(preds, labels)
                if batch_idx == 0:
                    vis_images = images
                    vis_labels = labels
                    vis_preds = preds
                
            loss_v = float(utils.from_torch(loss))

            if np.isnan(loss_v) or np.isinf(loss_v):
                return {"loss": loss_v,
                        "pred": utils.from_torch(preds),
                        "labels": utils.from_torch(labels)}
            
            mean_loss += loss_v
            optimizer.zero_grad()
            loss.backward()
            # optimizer optimizes the network parameters
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if torch.cuda.is_available() and iterations % 10 == 0:
                torch.cuda.empty_cache()
        
        # Create comprehensive training plots every 10 epochs
        if iterations % 10 == 0 and vis_images is not None:
            plot_dir = "./training_plots"
            utils.mkdir(plot_dir)
            
            # Get first sample from first batch (matching snake_dm)
            img_np = utils.from_torch(vis_images[0].cpu())[0]
            label_np = utils.from_torch(vis_labels[0].cpu())[0]
            pred_np = utils.from_torch(vis_preds[0].cpu())[0]
            
            # Check if snake-adjusted distance maps are available
            using_snake = (self.ours and iterations >= self.ours_start)
            has_snake_dm = using_snake and snake_adjusted_dmap is not None and snake_adjusted_dmap_initial is not None
            num_plots = 5 if has_snake_dm else 3
            
            fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 5))
            
            loss_type = "SnakeSimple (Width-Aware)" if using_snake else "MSE"
            fig.suptitle(f"Training Epoch {iterations} | Loss: {loss_type} = {mean_loss/len(self.dataloader):.3f}")
            
            # 1. Input Image
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            axes[0].imshow(img_norm, cmap='gray', origin='lower')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # 2. Ground Truth Distance Map
            im1 = axes[1].imshow(label_np, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
            axes[1].set_title(f'GT Signed DMap\n(min:{label_np.min():.1f}, max:{label_np.max():.1f})')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Dist')
            
            # 3. Prediction Distance Map
            im2 = axes[2].imshow(pred_np, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
            axes[2].set_title(f'Prediction\n(min:{pred_np.min():.1f}, max:{pred_np.max():.1f})')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Dist')
            
            # 4 & 5. Snake Distance Maps (Before and After Optimization)
            if has_snake_dm:
                snake_initial_np = utils.from_torch(snake_adjusted_dmap_initial[0].cpu())[0]
                im3 = axes[3].imshow(snake_initial_np, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
                axes[3].set_title(f'Snake BEFORE\n(min:{snake_initial_np.min():.1f}, max:{snake_initial_np.max():.1f})')
                axes[3].axis('off')
                plt.colorbar(im3, ax=axes[3], fraction=0.046, label='Dist')
                
                snake_final_np = utils.from_torch(snake_adjusted_dmap[0].cpu())[0]
                im4 = axes[4].imshow(snake_final_np, cmap='RdBu_r', origin='lower', vmin=-15, vmax=15)
                axes[4].set_title(f'Snake AFTER\n(min:{snake_final_np.min():.1f}, max:{snake_final_np.max():.1f})')
                axes[4].axis('off')
                plt.colorbar(im4, ax=axes[4], fraction=0.046, label='Dist')
            
            plt.tight_layout()
            plot_filename = os.path.join(plot_dir, f"training_epoch_{iterations:06d}.png")
            plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved training plot to {plot_filename}")

        return {"loss": float(mean_loss/len(self.dataloader))}
    
    
class Validation(object):

    def __init__(self, crop_size, margin_size, dataloader_val, out_channels, output_path):
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataloader_val = dataloader_val
        self.out_channels = out_channels
        self.output_path = output_path

        self.bestqual = 0
        self.bestcomp = 0
        self.bestcorr = 0

    def __call__(self, iteration, network, loss_function):

        losses = []
        preds = []
        scores = {"corr":[], "comp":[], "qual":[]}

        drive_output_path = "/content/drive/MyDrive/snake_model_outputs"

        network.train(False)
        with utils.torch_no_grad:
            for i, data_batch in enumerate(self.dataloader_val):
                image, label, original_shape = data_batch
                image  = image.cuda()
                label  = label.cuda()

                out_shape = (image.shape[0],self.out_channels,*image.shape[2:])
                pred = utils.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred = utils.process_in_chuncks(image, pred,
                                            lambda chunk: network(chunk),
                                            self.crop_size, self.margin_size)

                loss = loss_function(pred, label)
                loss_v = float(utils.from_torch(loss))
                losses.append(loss_v)

                pred_np = utils.from_torch(pred)[0]
                preds.append(pred_np)
                label_np = utils.from_torch(label)[0]
                
                # Debug: Print prediction statistics
                if i == 0:  # Only for first validation sample
                    pred_min, pred_max = pred_np.min(), pred_np.max()
                    pred_neg_count = (pred_np < 0).sum()
                    pred_pos_count = (pred_np > 0).sum()
                    logger.info(f"Validation - Pred range: [{pred_min:.2f}, {pred_max:.2f}], "
                              f"Negative pixels: {pred_neg_count}, Positive pixels: {pred_pos_count}")
                
                pred_mask = skeletonize((pred_np <= 5)[0].astype(np.uint8))
                label_mask = (label_np==0)

                corr, comp, qual = correctness_completeness_quality(pred_mask, label_mask, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # save preds
                pred_outs_path = os.path.join(drive_output_path, "output_valid")
                utils.mkdir(pred_outs_path)

                output_valid = os.path.join(self.output_path, "output_valid")
                utils.mkdir(output_valid)

                input_filename = os.path.join(output_valid, "val_input_{:03d}.npy".format(i))
                if not os.path.exists(input_filename):
                    np.save(input_filename, utils.from_torch(image)[0])

                gt_filename = os.path.join(output_valid, "val_gt_{:03d}.npy".format(i))
                if not os.path.exists(gt_filename):
                    np.save(gt_filename, label_np)

                pred_filename = os.path.join(pred_outs_path, "val_pred_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(pred_filename, pred_np)

                pred_mask_filename = os.path.join(pred_outs_path, "val_predmask_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(pred_mask_filename, pred_mask)
                
                # Create visualization plot
                plot_dir = "./validation_plots"
                utils.mkdir(plot_dir)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Validation Iter {iteration} - Sample {i} | Loss: {loss_v:.3f} | Qual: {qual:.3f}")
                
                # Normalize image for display
                img_display = utils.from_torch(image)[0][0]
                img_min, img_max = img_display.min(), img_display.max()
                if img_max > img_min:
                    img_display = (img_display - img_min) / (img_max - img_min)
                
                axes[0].imshow(img_display, cmap='gray', origin='lower')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(label_np[0], cmap='viridis', origin='lower')
                axes[1].set_title('Ground Truth Distance Map')
                axes[1].axis('off')
                
                axes[2].imshow(pred_np[0], cmap='viridis', origin='lower')
                axes[2].set_title('Prediction Distance Map')
                axes[2].axis('off')
                
                plt.tight_layout()
                plot_filename = os.path.join(plot_dir, f"val_plot_iter_{iteration:06d}_sample_{i:03d}.png")
                plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
                plt.close(fig)

        scores["qual"] = np.nan_to_num(scores["qual"])
        
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)

        if self.bestqual < qual_total:
            self.bestqual = qual_total
            for i in range(len(self.dataloader_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestqual.npy".format(i,iteration)), preds[i])
            utils.torch_save(os.path.join(self.output_path, "network_bestqual.pickle"),
                             network.state_dict())
        

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean qual: {:0.3f}".format(qual_total))
        logger.info("Best quality score is {}".format(self.bestqual))
        
        network.train(True)

        return {"loss": np.mean(losses),
                "mean_corr": corr_total,
                "mean_comp": comp_total,
                "mean_qual": qual_total,
                "scores": scores}
    
    
    
    