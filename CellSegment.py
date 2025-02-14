    def process_single_file(self, wsi_path, msk_path, output_dir):
        """Process a single whole-slide image and save the results.

        Args:
            wsi_path: path to input whole-slide image
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            output_dir: path where output will be saved

        """
        # TODO: customize universal file handler to sync the protocol
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape = np.array(self.chunk_shape)
        patch_input_shape = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        # path_split = pathlib.Path(self.presplit_dir + '/' + path_obj.stem + '.xml')
        # path_split = path_split if path_split.exists() else None
        path_split = None
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        start = time.perf_counter()
            
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext, path_presplit = path_split)
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        self.wsi_handler.prepare_reading(
            read_mag=self.proc_mag, cache_path="%s/src_wsi.npy" % self.cache_path
        )
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X
        print(msk_path)
        if msk_path is not None and os.path.isfile(msk_path):
            self.wsi_mask = cv2.imread(msk_path)
            self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
            self.wsi_mask[self.wsi_mask > 0] = 1
            print("use mask")
        else:
            log_info(
                "WARNING: No mask found, generating mask via thresholding at 1.25x!"
            )

            from skimage import morphology

            # simple method to extract tissue regions using intensity thresholding and morphological operations
            def simple_get_mask():
                scaled_wsi_mag = 1.25  # ! hard coded
                wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
                gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                # mask = morphology.remove_small_objects(
                #     mask == 0, min_size=16 * 16, connectivity=2
                # )
                # mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
                # mask = morphology.binary_dilation(mask, morphology.disk(16))
                # ! 2021-11-22, for 20x
                mask = morphology.remove_small_objects(
                    mask == 0, min_size=8 * 8, connectivity=2
                )
                mask = morphology.remove_small_holes(mask, area_threshold=64 * 64)
                mask = morphology.binary_dilation(mask, morphology.disk(8))
                return mask

            self.wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)
        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        # * declare holder for output
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        # TODO: dynamicalize this, retrieve from model?
        out_ch = 3 if self.method["model_args"]["nr_types"] is None else 4
        self.wsi_inst_info = {}
        # TODO: option to use entire RAM if users have too much available, would be faster than mmap
        self.wsi_inst_map = np.lib.format.open_memmap(
            "%s/pred_inst.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape),
            dtype=np.int32,
        )
        # 2021-11-18
        # self.wsi_inst_map = np.zeros(tuple(self.wsi_proc_shape), dtype=np.int32)
        # self.wsi_inst_map[:] = 0 # flush fill

        # warning, the value within this is uninitialized
        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )
        # 2021-11-03
        # self.wsi_pred_map = np.zeros(tuple(self.wsi_proc_shape) + (out_ch,), dtype=np.float32)
        # ! for debug
        # self.wsi_pred_map = np.load('%s/pred_map.npy' % self.cache_path, mmap_mode='r')
        end = time.perf_counter()
        log_info("Preparing Input Output Placement: {0}".format(end - start))

        # * raw prediction
        start = time.perf_counter()
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
            self.wsi_proc_shape,
            chunk_input_shape,
            patch_input_shape,
            patch_output_shape,
        )

        # get the raw prediction of HoVer-Net, given info of inference tiles and patches
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        log_info("Inference Time: {0}".format(end - start))

        # TODO: deal with error banding
        ##### * post processing
        ##### * done in 3 stages to ensure that nuclei at the boundaries are dealt with accordingly
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        # 3 sets of patches are extracted and are dealt with differently
        # tile_grid_info: central region of post processing tiles
        # tile_boundary_info: boundary region of post processing tiles
        # tile_cross_info: region at corners of post processing tiles
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set
        tile_grid_info = self.__select_valid_patches(tile_grid_info, False)
        tile_boundary_info = self.__select_valid_patches(tile_boundary_info, False)
        tile_cross_info = self.__select_valid_patches(tile_cross_info, False)

        ####################### * Callback can only receive 1 arg
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())
            for inst_id, inst_info in inst_info_dict.items():
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = pred_inst

            pbar.update()  # external
            return

        ####################### * Callback can only receive 1 arg
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
            # of the existing prediction map, and replace all nuclei within the region with newly predicted

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            # ! must get before the removal happened
            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())

            # * exclude ambiguous out from old prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_inst = self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ]
            roi_inst = np.copy(roi_inst)
            roi_edge = np.concatenate(
                [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
            )
            roi_boundary_inst_list = np.unique(roi_edge)[1:]  # exclude background
            roi_inner_inst_list = np.unique(roi_inst)[1:]
            roi_inner_inst_list = np.setdiff1d(
                roi_inner_inst_list, roi_boundary_inst_list, assume_unique=True
            )
            roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = roi_inst
            for inst_id in roi_inner_inst_list:
                self.wsi_inst_info.pop(inst_id, None)

            # * exclude unambiguous out from new prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_edge = pred_inst[roi_inst > 0]  # remove all overlap
            boundary_inst_list = np.unique(roi_edge)  # no background to exclude
            inner_inst_list = np.unique(pred_inst)[1:]
            inner_inst_list = np.setdiff1d(
                inner_inst_list, boundary_inst_list, assume_unique=True
            )
            pred_inst = _remove_inst(pred_inst, boundary_inst_list)

            # * proceed to overwrite
            for inst_id in inner_inst_list:
                # ! happen because we alrd skip thoses with wrong
                # ! contour (<3 points) within the postproc, so
                # ! sanity gate here
                if inst_id not in inst_info_dict:
                    log_info("Nuclei id=%d not in saved dict WRN1." % inst_id)
                    continue
                inst_info = inst_info_dict[inst_id]
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            pred_inst = roi_inst + pred_inst
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = pred_inst

            pbar.update()  # external
            return

        #######################
        pbar_creator = lambda x, y: tqdm.tqdm(
            desc=y, leave=True, total=int(len(x)), ncols=80, ascii=True, position=0
        )
        pbar = pbar_creator(tile_grid_info, "Post Proc Phase 1")
        # * must be in sequential ordering
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        pbar.close()

        pbar = pbar_creator(tile_boundary_info, "Post Proc Phase 2")
        self.__dispatch_post_processing(
            tile_boundary_info, post_proc_fixing_tile_callback
        )
        pbar.close()

        pbar = pbar_creator(tile_cross_info, "Post Proc Phase 3")
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        pbar.close()

        end = time.perf_counter()
        log_info("Total Post Proc Time: {0}".format(end - start))

        # ! cant possibly save the inst map at high res, too large
        start = time.perf_counter()
        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))
