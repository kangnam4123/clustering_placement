##
# @file   BasicPlace.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Base placement class
#

import os
import sys
import time
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import re
import numpy as np
import logging
import torch
import torch.nn as nn
import dreamplace.ops.move_boundary.move_boundary as move_boundary
import dreamplace.ops.hpwl.hpwl as hpwl
#import dreamplace.ops.rmst_wl.rmst_wl as rmst_wl
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
import dreamplace.ops.legality_check.legality_check as legality_check
import dreamplace.ops.draw_place.draw_place as draw_place
import dreamplace.ops.pin_pos.pin_pos as pin_pos
import dreamplace.ops.global_swap.global_swap as global_swap
import dreamplace.ops.k_reorder.k_reorder as k_reorder
import dreamplace.ops.independent_set_matching.independent_set_matching as independent_set_matching
import dreamplace.ops.pin_weight_sum.pin_weight_sum as pws
import dreamplace.ops.timing.timing as timing
import pdb
#내가 추가한거
import matplotlib
matplotlib.use('Agg')
import igraph as ig

class PlaceDataCollection(object):
    """
    @brief A wraper for all data tensors on device for building ops
    """
    def __init__(self, pos, params, placedb, device):
        """
        @brief initialization
        @param pos locations of cells
        @param params parameters
        @param placedb placement database
        @param device cpu or cuda
        """
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter
        self.pos = pos

        with torch.no_grad():
            # other tensors required to build ops

            self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)
            # original node size for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_node_size_x = self.node_size_x.clone()
                self.original_node_size_y = self.node_size_y.clone()

            self.pin_offset_x = torch.tensor(placedb.pin_offset_x,
                                             dtype=self.pos[0].dtype,
                                             device=device)
            self.pin_offset_y = torch.tensor(placedb.pin_offset_y,
                                             dtype=self.pos[0].dtype,
                                             device=device)
            # original pin offset for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_pin_offset_x = self.pin_offset_x.clone()
                self.original_pin_offset_y = self.pin_offset_y.clone()

            self.target_density = torch.empty(1,
                                              dtype=self.pos[0].dtype,
                                              device=device)
            self.target_density.data.fill_(params.target_density)

            # detect movable macros and scale down the density to avoid halos
            # I use a heuristic that cells whose areas are 10x of the mean area will be regarded movable macros in global placement
            self.node_areas = self.node_size_x * self.node_size_y
            if self.target_density < 2:
                mean_area = self.node_areas[:placedb.num_movable_nodes].mean(
                ).mul_(1)
                row_height = self.node_size_y[:placedb.num_movable_nodes].min(
                ).mul_(1)
                self.movable_macro_mask = (
                    self.node_areas[:placedb.num_movable_nodes] > mean_area
                ) & (self.node_size_y[:placedb.num_movable_nodes] > row_height)
            else:  # no movable macros
                self.movable_macro_mask = None

            self.pin2node_map = torch.from_numpy(
                placedb.pin2node_map).to(device)
            self.flat_node2pin_map = torch.from_numpy(
                placedb.flat_node2pin_map).to(device)
            self.flat_node2pin_start_map = torch.from_numpy(
                placedb.flat_node2pin_start_map).to(device)
            # number of pins for each cell
            self.pin_weights = (self.flat_node2pin_start_map[1:] -
                                self.flat_node2pin_start_map[:-1]).to(
                                    self.node_size_x.dtype)

            self.unit_pin_capacity = torch.empty(1,
                                                 dtype=self.pos[0].dtype,
                                                 device=device)
            self.unit_pin_capacity.data.fill_(params.unit_pin_capacity)
            if params.routability_opt_flag:
                unit_pin_capacity = self.pin_weights[:placedb.
                                                     num_movable_nodes] / self.node_areas[:placedb
                                                                                          .
                                                                                          num_movable_nodes]
                avg_pin_capacity = unit_pin_capacity.mean(
                ) * self.target_density
                # min(computed, params.unit_pin_capacity)
                self.unit_pin_capacity = avg_pin_capacity.clamp_(
                    max=params.unit_pin_capacity)
                logging.info("unit_pin_capacity = %g" %
                             (self.unit_pin_capacity))

            # routing information
            # project initial routing utilization map to one layer
            self.initial_horizontal_utilization_map = None
            self.initial_vertical_utilization_map = None
            if params.routability_opt_flag and placedb.initial_horizontal_demand_map is not None:
                self.initial_horizontal_utilization_map = torch.from_numpy(
                    placedb.initial_horizontal_demand_map).to(device).div_(
                        placedb.routing_grid_size_y *
                        placedb.unit_horizontal_capacity)
                self.initial_vertical_utilization_map = torch.from_numpy(
                    placedb.initial_vertical_demand_map).to(device).div_(
                        placedb.routing_grid_size_x *
                        placedb.unit_vertical_capacity)

            self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
            self.flat_net2pin_map = torch.from_numpy(
                placedb.flat_net2pin_map).to(device)
            self.flat_net2pin_start_map = torch.from_numpy(
                placedb.flat_net2pin_start_map).to(device)

            self.net_weights = torch.from_numpy(placedb.net_weights).to(device)

            # regions
            self.flat_region_boxes = torch.from_numpy(
                placedb.flat_region_boxes).to(device)
            self.flat_region_boxes_start = torch.from_numpy(
                placedb.flat_region_boxes_start).to(device)
            self.node2fence_region_map = torch.from_numpy(
                placedb.node2fence_region_map).to(device)
            if len(placedb.regions) > 0:
                # This is for multi-electric potential and legalization
                # boxes defined as left-bottm point and top-right point
                self.virtual_macro_fence_region = [torch.from_numpy(region).to(device) for region in placedb.virtual_macro_fence_region]
                ## this is for overflow op
                self.total_movable_node_area_fence_region = torch.from_numpy(placedb.total_movable_node_area_fence_region).to(device)
                ## this is for gamma update
                self.num_movable_nodes_fence_region = torch.from_numpy(placedb.num_movable_nodes_fence_region).to(device)
                ## this is not used yet
                self.num_filler_nodes_fence_region = torch.from_numpy(placedb.num_filler_nodes_fence_region).to(device)

            net_degrees = self.flat_net2pin_start_map[1:] - self.flat_net2pin_start_map[:-1]
            self.net_mask_all = (2 <= net_degrees).to(torch.uint8) # all valid nets included
            self.net_mask_ignore_large_degrees = torch.logical_and(
                    self.net_mask_all, 
                    net_degrees < params.ignore_net_degree).to(torch.uint8) # nets with large degrees are ignored

            # number of pins for each node
            num_pins_in_nodes = np.zeros(placedb.num_nodes)
            for i in range(placedb.num_physical_nodes):
                num_pins_in_nodes[i] = len(placedb.node2pin_map[i])
            self.num_pins_in_nodes = torch.tensor(num_pins_in_nodes,
                                                  dtype=self.pos[0].dtype,
                                                  device=device)

            # sum of pin weights for each node.
            sum_pin_weights_in_nodes = np.zeros(placedb.num_nodes)
            self.sum_pin_weights_in_nodes = \
                torch.tensor(sum_pin_weights_in_nodes,
                             dtype=self.net_weights.dtype,
                             device="cpu")

            # avoid computing gradient for fixed macros
            # 1 is for fixed macros
            self.pin_mask_ignore_fixed_macros = (self.pin2node_map >=
                                                 placedb.num_movable_nodes)

            # sort nodes by size, return their sorted indices, designed for memory coalesce in electrical force
            movable_size_x = self.node_size_x[:placedb.num_movable_nodes]
            _, self.sorted_node_map = torch.sort(movable_size_x)
            self.sorted_node_map = self.sorted_node_map.to(torch.int32)
            # self.sorted_node_map = torch.arange(0, placedb.num_movable_nodes, dtype=torch.int32, device=device)

            # logging.debug(self.node_size_x[placedb.num_movable_nodes//2 :placedb.num_movable_nodes//2+20])
            # logging.debug(self.sorted_node_map[placedb.num_movable_nodes//2 :placedb.num_movable_nodes//2+20])
            # logging.debug(self.node_size_x[self.sorted_node_map[0: 10].long()])
            # logging.debug(self.node_size_x[self.sorted_node_map[-10:].long()])

    def bin_center_x_padded(self, placedb, padding, num_bins_x):
        """
        @brief compute array of bin center horizontal coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        xl = placedb.xl - padding * bin_size_x
        xh = placedb.xh + padding * bin_size_x
        bin_center_x = torch.from_numpy(
            placedb.bin_centers(xl, xh, bin_size_x)).to(self.device)
        return bin_center_x

    def bin_center_y_padded(self, placedb, padding, num_bins_y):
        """
        @brief compute array of bin center vertical coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y
        yl = placedb.yl - padding * bin_size_y
        yh = placedb.yh + padding * bin_size_y
        bin_center_y = torch.from_numpy(
            placedb.bin_centers(yl, yh, bin_size_y)).to(self.device)
        return bin_center_y


class PlaceOpCollection(object):
    """
    @brief A wrapper for all ops
    """
    def __init__(self):
        """
        @brief initialization
        """
        self.pin_pos_op = None
        self.move_boundary_op = None
        self.hpwl_op = None
        self.rmst_wl_op = None
        self.density_overflow_op = None
        self.legality_check_op = None
        self.legalize_op = None
        self.detailed_place_op = None
        self.wirelength_op = None
        self.update_gamma_op = None
        self.density_op = None
        self.update_density_weight_op = None
        self.precondition_op = None
        self.noise_op = None
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None
        self.nctugr_congestion_map_op = None
        self.adjust_node_area_op = None


class BasicPlace(nn.Module):
    """
    @brief Base placement class.
    All placement engines should be derived from this class.
    """
    def __init__(self, params, placedb, timer):
        """
        @brief initialization
        @param params parameter
        @param placedb placement database
        @param timer the timing analysis engine
        """
        torch.manual_seed(params.random_seed)
        super(BasicPlace, self).__init__()

        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)
        # x position
        self.init_pos[0:placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
            logging.info(
                "move cells to the center of layout with random noise")
            self.init_pos[0:placedb.num_movable_nodes] = np.random.normal(
                loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
                scale=(placedb.xh - placedb.xl) * 0.001,
                size=placedb.num_movable_nodes)

        # y position
        self.init_pos[placedb.num_nodes:placedb.num_nodes +
                      placedb.num_physical_nodes] = placedb.node_y
        if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
            self.init_pos[placedb.num_nodes:placedb.num_nodes +
                          placedb.num_movable_nodes] = np.random.normal(
                              loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
                              scale=(placedb.yh - placedb.yl) * 0.001,
                              size=placedb.num_movable_nodes)

        # clustering_initial placement position
        if params.global_place_flag and params.clustering_initial_position:
            #clustering result 만들기 - 혹은 clustering result 불러와서  배치 이후 위치만 옮기기
            logging.info(
                "move cells to the clustering based placement result")
            initiallocation = self.clustering(placedb)
            self.init_pos[0:placedb.num_movable_nodes] = initiallocation[0:placedb.num_movable_nodes]
            self.init_pos[placedb.num_nodes:placedb.num_nodes +placedb.num_movable_nodes] = initiallocation[placedb.num_movable_nodes:placedb.num_movable_nodes*2]
            
        if placedb.num_filler_nodes:  # uniformly distribute filler cells in the layout
            if len(placedb.regions) > 0:
                ### uniformly spread fillers in fence region
                ### for cells in the fence region
                for i, region in enumerate(placedb.regions):
                    filler_beg, filler_end = placedb.filler_start_map[i : i + 2]
                    subregion_areas = (region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1])
                    total_area = np.sum(subregion_areas)
                    subregion_area_ratio = subregion_areas / total_area
                    subregion_num_filler = np.round((filler_end - filler_beg) * subregion_area_ratio)
                    subregion_num_filler[-1] = (filler_end - filler_beg) - np.sum(subregion_num_filler[:-1])
                    subregion_num_filler_start_map = np.concatenate(
                        [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                    ).astype(np.int32)
                    for j, subregion in enumerate(region):
                        sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[j : j + 2]
                        self.init_pos[
                            placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg : placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[0],
                            high=subregion[2] - placedb.filler_size_x_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )
                        self.init_pos[
                            placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg : placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[1],
                            high=subregion[3] - placedb.filler_size_y_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )

                ### for cells outside fence region
                filler_beg, filler_end = placedb.filler_start_map[-2:]
                self.init_pos[
                    placedb.num_physical_nodes + filler_beg : placedb.num_physical_nodes + filler_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.filler_size_x_fence_region[-1],
                    size=filler_end - filler_beg,
                )
                self.init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_beg : placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.filler_size_y_fence_region[-1],
                    size=filler_end - filler_beg,
                )

            else:
                self.init_pos[placedb.num_physical_nodes : placedb.num_nodes] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.node_size_x[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )
                self.init_pos[
                    placedb.num_nodes + placedb.num_physical_nodes : placedb.num_nodes * 2
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.node_size_y[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )

        logging.debug("prepare init_pos takes %.2f seconds" %
                      (time.time() - tt))

        self.device = torch.device("cuda" if params.gpu else "cpu")

        # position should be parameter
        # must be defined in BasicPlace
        tt = time.time()
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))])
        logging.debug("build pos takes %.2f seconds" % (time.time() - tt))
        # shared data on device for building ops
        # I do not want to construct the data from placedb again and again for each op
        tt = time.time()
        self.data_collections = PlaceDataCollection(self.pos, params, placedb,
                                                    self.device)
        logging.debug("build data_collections takes %.2f seconds" %
                      (time.time() - tt))

        # similarly I wrap all ops
        tt = time.time()
        self.op_collections = PlaceOpCollection()
        logging.debug("build op_collections takes %.2f seconds" %
                      (time.time() - tt))

        tt = time.time()
        # position to pin position
        self.op_collections.pin_pos_op = self.build_pin_pos(
            params, placedb, self.data_collections, self.device)
        # bound nodes to layout region
        self.op_collections.move_boundary_op = self.build_move_boundary(
            params, placedb, self.data_collections, self.device)
        # hpwl and density overflow ops for evaluation
        self.op_collections.hpwl_op = self.build_hpwl(
            params, placedb, self.data_collections,
            self.op_collections.pin_pos_op, self.device)
        self.op_collections.pws_op = self.build_pws(placedb, self.data_collections)
        # rectilinear minimum steiner tree wirelength from flute
        # can only be called once
        #self.op_collections.rmst_wl_op = self.build_rmst_wl(params, placedb, self.op_collections.pin_pos_op, torch.device("cpu"))
        self.op_collections.timing_op = self.build_timing_op(params, placedb, timer)
        # legality check
        self.op_collections.legality_check_op = self.build_legality_check(
            params, placedb, self.data_collections, self.device)
        # legalization
        if len(placedb.regions) > 0:
            self.op_collections.legalize_op, self.op_collections.individual_legalize_op = self.build_multi_fence_region_legalization(
            params, placedb, self.data_collections, self.device)
        else:
            self.op_collections.legalize_op = self.build_legalization(
            params, placedb, self.data_collections, self.device)
        # detailed placement
        self.op_collections.detailed_place_op = self.build_detailed_placement(
            params, placedb, self.data_collections, self.device)
        # draw placement
        self.op_collections.draw_place_op = self.build_draw_placement(
            params, placedb)

        # flag for rmst_wl_op
        # can only read once
        self.read_lut_flag = True

        logging.debug("build BasicPlace ops takes %.2f seconds" %
                      (time.time() - tt))

    def __call__(self, params, placedb):
        """
        @brief Solve placement.
        placeholder for derived classes.
        @param params parameters
        @param placedb placement database
        """
        pass

    def build_pin_pos(self, params, placedb, data_collections, device):
        """
        @brief sum up the pins for each cell
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # Yibo: I found CPU version of this is super slow, more than 2s for ISPD2005 bigblue4 with 10 threads.
        # So I implemented a custom CPU version, which is around 20ms
        #pin2node_map = data_collections.pin2node_map.long()
        #def build_pin_pos_op(pos):
        #    pin_x = data_collections.pin_offset_x.add(torch.index_select(pos[0:placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_y = data_collections.pin_offset_y.add(torch.index_select(pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_pos = torch.cat([pin_x, pin_y], dim=0)

        #    return pin_pos
        #return build_pin_pos_op

        return pin_pos.PinPos(
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            pin2node_map=data_collections.pin2node_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            num_physical_nodes=placedb.num_physical_nodes,
            algorithm="node-by-node")

    def build_move_boundary(self, params, placedb, data_collections, device):
        """
        @brief bound nodes into layout region
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return move_boundary.MoveBoundary(
            data_collections.node_size_x,
            data_collections.node_size_y,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes)

    def build_hpwl(self, params, placedb, data_collections, pin_pos_op,
                   device):
        """
        @brief compute half-perimeter wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """

        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_all,
            algorithm='net-by-net')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        return build_wirelength_op
    
    def build_pws(self, placedb, data_collections):
        """
        @brief accumulate pin weights of a node
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        # CPU version by default...
        pws_op = pws.PinWeightSum(
            flat_nodepin=data_collections.flat_node2pin_map,
            nodepin_start=data_collections.flat_node2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            num_nodes=placedb.num_nodes,
            algorithm='node-by-node')

        return pws_op

    def build_rmst_wl(self, params, placedb, pin_pos_op, device):
        """
        @brief compute rectilinear minimum spanning tree wirelength with flute
        @param params parameters
        @param placedb placement database
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """
        # wirelength cost

        POWVFILE = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../thirdparty/NCTUgr.ICCAD2012/POWV9.dat"))
        POSTFILE = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "../../thirdparty/NCTUgr.ICCAD2012/POST9.dat"))
        logging.info("POWVFILE = %s" % (POWVFILE))
        logging.info("POSTFILE = %s" % (POSTFILE))
        wirelength_for_pin_op = rmst_wl.RMSTWL(
            flat_netpin=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            netpin_start=torch.from_numpy(
                placedb.flat_net2pin_start_map).to(device),
            ignore_net_degree=params.ignore_net_degree,
            POWVFILE=POWVFILE,
            POSTFILE=POSTFILE)

        # wirelength for position
        def build_wirelength_op(pos):
            pin_pos = pin_pos_op(pos)
            wls = wirelength_for_pin_op(pin_pos.clone().cpu(),
                                        self.read_lut_flag)
            self.read_lut_flag = False
            return wls

        return build_wirelength_op

    def build_timing_op(self, params, placedb, timer=None):
        """
        @brief build the operator for timing analysis and feedbacks.
        @param placedb the placement database
        @param timer the timer object used in timing-driven mode
        """
        return timing.TimingOpt(
            timer, # The timer should be at the same level as placedb.
            placedb.net_names, # The net names are required by OpenTimer.
            placedb.pin_names, # The pin names are required by OpenTimer.
            placedb.flat_net2pin_map,
            placedb.flat_net2pin_start_map,
            placedb.net_name2id_map,
            placedb.pin_name2id_map,
            placedb.pin2node_map,
            placedb.pin_offset_x,
            placedb.pin_offset_y,
            placedb.net_criticality,
            placedb.net_criticality_deltas,
            placedb.net_weights,
            placedb.net_weight_deltas,
            wire_resistance_per_micron=params.wire_resistance_per_micron,
            wire_capacitance_per_micron=params.wire_capacitance_per_micron,
            net_weighting_scheme=params.net_weighting_scheme,
            momentum_decay_factor=params.momentum_decay_factor,
            scale_factor=params.scale_factor,
            lef_unit=placedb.rawdb.lefUnit(),
            def_unit=placedb.rawdb.defUnit(),
            ignore_net_degree=params.ignore_net_degree)

    def build_legality_check(self, params, placedb, data_collections, device):
        """
        @brief legality check
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return legality_check.LegalityCheck(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            scale_factor=params.scale_factor,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes)

    def build_legalization(self, params, placedb, data_collections, device):
        """
        @brief legalization
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # for movable macro legalization
        # the number of bins control the search granularity
        ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)
        # for standard cell legalization
        # legalize_alg = mg_legalize.MGLegalize
        legalize_alg = greedy_legalize.GreedyLegalize
        gl = legalize_alg(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            #num_bins_x=64, num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            #num_bins_x=64, num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes)

        def build_legalization_op(pos):
            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            pos2 = gl(pos1, pos1)
            legal = self.op_collections.legality_check_op(pos2)
            if not legal:
                logging.error("legality check failed in greedy legalization, " \
                    "return illegal results after greedy legalization.")
                return pos2
            pos3 = al(pos1, pos2)
            legal = self.op_collections.legality_check_op(pos3)
            if not legal:
                logging.error("legality check failed in abacus legalization, " \
                    "return legal results after greedy legalization.")
                return pos2
            return pos3

        return build_legalization_op

    def build_multi_fence_region_legalization(self, params, placedb, data_collections, device):
        legal_ops = [self.build_fence_region_legalization(region_id, params, placedb, data_collections, device) for region_id in range(len(placedb.regions)+1)]

        pos_ml_list = []
        pos_gl_list = []
        def build_legalization_op(pos):
            for i in range(len(placedb.regions)+1):
                pos, pos_ml, pos_gl = legal_ops[i][0](pos)
                pos_ml_list.append(pos_ml)
                pos_gl_list.append(pos_gl)
            legal = self.op_collections.legality_check_op(pos)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos
            else:
                ### start abacus legalizer
                for i in range(len(placedb.regions)+1):
                    pos = legal_ops[i][1](pos, pos_ml_list[i], pos_gl_list[i])
            return pos

        def build_individual_legalization_ops(pos, region_id):
            pos = legal_ops[region_id][0](pos)[0]
            return pos

        return build_legalization_op, build_individual_legalization_ops

    def build_fence_region_legalization(self, region_id, params, placedb, data_collections, device):
        ### reconstruct node size
        ### extract necessary nodes in the electric field and insert virtual macros to replace fence region
        num_nodes = placedb.num_nodes
        num_movable_nodes = placedb.num_movable_nodes
        num_filler_nodes = placedb.num_filler_nodes
        num_terminals = placedb.num_terminals
        num_terminal_NIs = placedb.num_terminal_NIs
        if region_id < len(placedb.regions):
            fence_region_mask = data_collections.node2fence_region_map[:num_movable_nodes] == region_id
        else:
            fence_region_mask = data_collections.node2fence_region_map[:num_movable_nodes] >= len(placedb.regions)

        virtual_macros = data_collections.virtual_macro_fence_region[region_id]
        virtual_macros_center_x = (virtual_macros[:,2] + virtual_macros[:,0]) / 2
        virtual_macros_center_y = (virtual_macros[:,3] + virtual_macros[:,1]) / 2
        virtual_macros_size_x = (virtual_macros[:,2]-virtual_macros[:,0]).clamp(min=30)

        virtual_macros_size_y = (virtual_macros[:,3]-virtual_macros[:,1]).clamp(min=30)
        virtual_macros[:, 0] = virtual_macros_center_x - virtual_macros_size_x / 2
        virtual_macros[:, 1] = virtual_macros_center_y - virtual_macros_size_y / 2
        virtual_macros_pos = virtual_macros[:,0:2].t().contiguous()

        ### node size
        node_size_x, node_size_y = data_collections.node_size_x, data_collections.node_size_y
        filler_beg, filler_end = placedb.filler_start_map[region_id:region_id+2]
        node_size_x = torch.cat([node_size_x[:num_movable_nodes][fence_region_mask], ## movable
                                node_size_x[num_movable_nodes:num_movable_nodes+num_terminals], ## terminals
                                virtual_macros_size_x, ## virtual macros
                                node_size_x[num_movable_nodes + num_terminals:num_movable_nodes + num_terminals + num_terminal_NIs], ## terminal NIs
                                node_size_x[num_nodes-num_filler_nodes + filler_beg:num_nodes-num_filler_nodes + filler_end] ## fillers
                                ], 0)
        node_size_y = torch.cat([node_size_y[:num_movable_nodes][fence_region_mask], ## movable
                                node_size_y[num_movable_nodes:num_movable_nodes + num_terminals], ## terminals
                                virtual_macros_size_y, ## virtual macros
                                node_size_y[num_movable_nodes + num_terminals:num_movable_nodes + num_terminals + num_terminal_NIs], ## terminal NIs
                                node_size_y[num_nodes - num_filler_nodes + filler_beg:num_nodes - num_filler_nodes + filler_end] ## fillers
                                ], 0)

        ### num pins in nodes
        ### 0 for virtual macros and fillers
        num_pins_in_nodes = data_collections.num_pins_in_nodes
        num_pins_in_nodes = torch.cat([num_pins_in_nodes[:num_movable_nodes][fence_region_mask], ## movable
                                       num_pins_in_nodes[num_movable_nodes:num_movable_nodes + num_terminals], ## terminals
                                       torch.zeros(virtual_macros_size_x.size(0), dtype=num_pins_in_nodes.dtype, device=device), ## virtual macros
                                       num_pins_in_nodes[num_movable_nodes + num_terminals:num_movable_nodes + num_terminals + num_terminal_NIs], ## terminal NIs
                                       num_pins_in_nodes[num_nodes - num_filler_nodes + filler_beg:num_nodes - num_filler_nodes + filler_end] ## fillers
                                       ], 0)
        ## num movable nodes and num filler nodes
        num_movable_nodes_fence_region = fence_region_mask.long().sum().item()
        num_filler_nodes_fence_region = filler_end - filler_beg
        num_terminals_fence_region = num_terminals + virtual_macros_size_x.size(0)
        assert node_size_x.size(0) == node_size_y.size(0) == num_movable_nodes_fence_region + num_terminals_fence_region + num_terminal_NIs + num_filler_nodes_fence_region

        ### flat region boxes
        flat_region_boxes = torch.tensor([], device=node_size_x.device, dtype=data_collections.flat_region_boxes.dtype)
        ### flat region boxes start
        flat_region_boxes_start = torch.tensor([0], device=node_size_x.device, dtype=data_collections.flat_region_boxes_start.dtype)
        ### node2fence region map: movable + terminal
        node2fence_region_map = torch.zeros(num_movable_nodes_fence_region + num_terminals_fence_region, dtype=data_collections.node2fence_region_map.dtype, device=node_size_x.device).fill_(data_collections.node2fence_region_map.max().item())

        ml = macro_legalize.MacroLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=params.num_bins_x,
            num_bins_y=params.num_bins_y,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region)

        gl = greedy_legalize.GreedyLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            #num_bins_x=64, num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region)
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region)

        def build_greedy_legalization_op(pos):
            ### reconstruct pos for fence region
            pos_total = pos.data.clone()
            pos = pos.view(2, -1)
            pos = torch.cat([pos[:, :num_movable_nodes][:, fence_region_mask], ## movable
                            pos[:, num_movable_nodes : num_movable_nodes + num_terminals], ## terminals
                            virtual_macros_pos, ## virtual macros
                            pos[:, num_movable_nodes + num_terminals:num_movable_nodes + num_terminals + num_terminal_NIs], ## terminal NIs
                            pos[:, num_nodes - num_filler_nodes + filler_beg : num_nodes - num_filler_nodes + filler_end] ## fillers
                            ], 1).view(-1).contiguous()
            assert pos.size(0) == 2 * node_size_x.size(0)

            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            result = gl(pos1, pos1)
            ## commit legal solution for movable cells in fence region
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(fence_region_mask, result[0, :num_movable_nodes_fence_region])
            pos_total[1, :num_movable_nodes].masked_scatter_(fence_region_mask, result[1, :num_movable_nodes_fence_region])
            pos_total = pos_total.view(-1).contiguous()
            result = result.view(-1).contiguous()
            return pos_total, pos1, result

        def build_abacus_legalization_op(pos_total, pos_ref, pos):
            result = al(pos_ref, pos)
            ### commit abacus results to pos_total
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(fence_region_mask, result[0, :num_movable_nodes_fence_region])
            pos_total[1, :num_movable_nodes].masked_scatter_(fence_region_mask, result[1, :num_movable_nodes_fence_region])
            pos_total = pos_total.view(-1).contiguous()
            return pos_total

        return build_greedy_legalization_op, build_abacus_legalization_op

    def build_detailed_placement(self, params, placedb, data_collections,
                                 device):
        """
        @brief detailed placement consisting of global swap and independent set matching
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        gs = global_swap.GlobalSwap(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x // 2,
            num_bins_y=placedb.num_bins_y // 2,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=256,
            max_iters=2,
            algorithm='concurrent')
        kr = k_reorder.KReorder(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            K=4,
            max_iters=2)
        ism = independent_set_matching.IndependentSetMatching(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            site_width=placedb.site_width,
            row_height=placedb.row_height,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=2048,
            set_size=128,
            max_iters=50,
            algorithm='concurrent')

        # wirelength for position
        def build_detailed_placement_op(pos):
            logging.info("Start ABCDPlace for refinement")

            if placedb.num_movable_nodes < 2: 
                logging.info("Too few movable cells, skip detailed placement")
                return pos 

            pos1 = pos
            legal = self.op_collections.legality_check_op(pos1)
            logging.info("ABCDPlace input legal flag = %d" %
                         (legal))
            if not legal:
                return pos1

            # integer factorization to prime numbers
            def prime_factorization(num):
                lt = []
                while num != 1:
                    for i in range(2, int(num+1)):
                        if num % i == 0:  # i is a prime factor
                            lt.append(i)
                            num = num / i # get the quotient for further factorization
                            break
                return lt

            # compute the scale factor for detailed placement
            # as the algorithms prefer integer coordinate systems
            scale_factor = params.scale_factor
            if params.scale_factor != 1.0:
                inv_scale_factor = int(round(1.0 / params.scale_factor))
                prime_factors = prime_factorization(inv_scale_factor)
                target_inv_scale_factor = 1
                for factor in prime_factors:
                    if factor != 2 and factor != 5:
                        target_inv_scale_factor = inv_scale_factor
                        break
                scale_factor = 1.0 / target_inv_scale_factor
                logging.info("Deriving from system scale factor %g (1/%d)" % (params.scale_factor, inv_scale_factor))
                logging.info("Use scale factor %g (1/%d) for detailed placement" % (scale_factor, target_inv_scale_factor))

            for i in range(1):
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = ism(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Independent set matching legal flag = %d" %
                             (legal))
                if not legal:
                    return pos1
                pos1 = gs(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Global swap legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
            return pos1

        return build_detailed_placement_op

    def build_draw_placement(self, params, placedb):
        """
        @brief plot placement
        @param params parameters
        @param placedb placement database
        """
        return draw_place.DrawPlace(placedb)

    def validate(self, placedb, pos, iteration):
        """
        @brief validate placement
        @param placedb placement database
        @param pos locations of cells
        @param iteration optimization step
        """
        pos = torch.from_numpy(pos).to(self.device)
        hpwl = self.op_collections.hpwl_op(pos)
        #rmst_wls = self.rmst_wl_op(pos)
        #rmst_wl = rmst_wls.sum()
        overflow, max_density = self.op_collections.density_overflow_op(pos)

        #return hpwl, rmst_wl, overflow, max_density
        return hpwl, overflow, max_density

    def plot(self, params, placedb, iteration, pos):
        """
        @brief plot layout
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        """
        tt = time.time()
        path = "%s/%s" % (params.result_dir, params.design_name())
        figname = "%s/plot/iter%s.png" % (path, '{:04}'.format(iteration))
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        logging.info("plotting to %s takes %.3f seconds" %
                     (figname, time.time() - tt))

    def dump(self, params, placedb, pos, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "wb") as f:
            pickle.dump(
                (self.data_collections.node_size_x.cpu(),
                 self.data_collections.node_size_y.cpu(),
                 self.data_collections.flat_net2pin_map.cpu(),
                 self.data_collections.flat_net2pin_start_map.cpu(),
                 self.data_collections.pin2net_map.cpu(),
                 self.data_collections.flat_node2pin_map.cpu(),
                 self.data_collections.flat_node2pin_start_map.cpu(),
                 self.data_collections.pin2node_map.cpu(),
                 self.data_collections.pin_offset_x.cpu(),
                 self.data_collections.pin_offset_y.cpu(),
                 self.data_collections.net_mask_ignore_large_degrees.cpu(),
                 placedb.xl, placedb.yl, placedb.xh, placedb.yh,
                 placedb.site_width, placedb.row_height, placedb.num_bins_x,
                 placedb.num_bins_y, placedb.num_movable_nodes,
                 placedb.num_terminal_NIs, placedb.num_filler_nodes, pos), f)

    def load(self, params, placedb, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
            self.data_collections.node_size_x.data = data[0].data.to(
                self.device)
            self.data_collections.node_size_y.data = data[1].data.to(
                self.device)
            self.data_collections.flat_net2pin_map.data = data[2].data.to(
                self.device)
            self.data_collections.flat_net2pin_start_map.data = data[
                3].data.to(self.device)
            self.data_collections.pin2net_map.data = data[4].data.to(
                self.device)
            self.data_collections.flat_node2pin_map.data = data[5].data.to(
                self.device)
            self.data_collections.flat_node2pin_start_map.data = data[
                6].data.to(self.device)
            self.data_collections.pin2node_map.data = data[7].data.to(
                self.device)
            self.data_collections.pin_offset_x.data = data[8].data.to(
                self.device)
            self.data_collections.pin_offset_y.data = data[9].data.to(
                self.device)
            self.data_collections.net_mask_ignore_large_degrees.data = data[
                10].data.to(self.device)
            placedb.xl = data[11]
            placedb.yl = data[12]
            placedb.xh = data[13]
            placedb.yh = data[14]
            placedb.site_width = data[15]
            placedb.row_height = data[16]
            placedb.num_bins_x = data[17]
            placedb.num_bins_y = data[18]
            num_movable_nodes = data[19]
            num_nodes = data[0].numel()
            placedb.num_terminal_NIs = data[20]
            placedb.num_filler_nodes = data[21]
            placedb.num_physical_nodes = num_nodes - placedb.num_filler_nodes
            placedb.num_terminals = placedb.num_physical_nodes - placedb.num_terminal_NIs - num_movable_nodes
            self.data_collections.pos[0].data = data[22].data.to(self.device)
    
    def name2id_map2str(m):
        id2name_map = [None]*len(m)
        for k in m.keys():
            id2name_map[m[k]] = k
        content = ""
        for i in range(len(m)):
            if i:
                content += ", "
            content += "%s : %d" % (id2name_map[i], i)
        return "{%s}" % (content)

    def array2str(a):
        content = ""
        for v in a:
            if content:
                content += ", "
            content += "%s" % (v)
        return "[%s]" % (content)

    def clustering(self, placedb):

        tt = time.time()
        content = ""
        content += "num_physical_nodes = %s\n" % (placedb.num_physical_nodes)
        content += "num_movable_nodes = %s\n" % (placedb.num_movable_nodes)
        content += "num_fixed_macros = %s\n" % (placedb.num_terminals)
        #content += "node_name2id_map = %s\n" % (name2id_map2str(placedb.node_name2id_map))
        #content += "node_names = %s\n" % (array2str(placedb.node_names))
        #content += "node_x = %s\n" % (placedb.node_x)
        #content += "node_y = %s\n" % (placedb.node_y)
        #content += "node_orient = %s\n" % (array2str(placedb.node_orient))
        #content += "node_size_y = %s\n" % (placedb.node_size_y)
        #content += "pin_direct = %s\n" % (array2str(placedb.pin_direct))
        #content += "pin_offset_x = %s\n" % (placedb.pin_offset_x)
        #content += "pin_offset_y = %s\n" % (placedb.pin_offset_y)
        #content += "net_name2id_map = %s\n" % (name2id_map2str(placedb.net_name2id_map))
        #content += "net_weights = %s\n" % (placedb.net_weights)
        #content += "net_names = %s\n" % (array2str(placedb.net_names))
        #content += "net2pin_map = %s\n" % (placedb.net2pin_map)
        #content += "flat_net2pin_map = %s\n" % (placedb.flat_net2pin_map)
        #content += "flat_net2pin_start_map = %s\n" % (placedb.flat_net2pin_start_map)
        #content += "node2pin_map = %s\n" % (placedb.node2pin_map)
        #content += "flat_node2pin_map = %s\n" % (placedb.flat_node2pin_map)
        #content += "flat_node2pin_start_map = %s\n" % (placedb.flat_node2pin_start_map)
        #content += "pin2node_map = %s\n" % (placedb.pin2node_map)
        #content += "pin_name2id_map = %s\n" % (name2id_map2str(placedb.pin_name2id_map))
        #content += "pin2net_map = %s\n" % (placedb.pin2net_map)
        #content += "rows = %s\n" % (placedb.rows)
        content += "xl = %s\n" % (placedb.xl)
        content += "yl = %s\n" % (placedb.yl)
        content += "xh = %s\n" % (placedb.xh)
        content += "yh = %s\n" % (placedb.yh)
        #content += "row_height = %s\n" % (placedb.row_height)
        #content += "site_width = %s\n" % (placedb.site_width)
        #content += "num_movable_pins = %s\n" % (placedb.num_movable_pins)

        #Macro cell을 인지해서(height가 몇배 이상이면 macro cell) cell list를 만들어서 macro cell이면 1로 만들기
        #db에 있는 data 쪼개서 graph넣어줄 수 있는 형태로 바꾸기
        isMacro = [0] * placedb.num_movable_nodes
        stdheight = min(placedb.node_size_y[0:placedb.num_movable_nodes])
        macronum = 0
        for i in range(placedb.num_movable_nodes):
            if(placedb.node_size_y[i] > stdheight):
                isMacro[i] = 1
                macronum +=1
                
        #Macro cell과 연결되어 있는 cell들은 1로 표시하기(Macro cell 중복돼도 괜찮)
        isMacro1 = [0] * placedb.num_movable_nodes
        for index, value in enumerate(isMacro):
            if value ==1:
                pins = placedb.node2pin_map[index]
                for pin in pins:
                    pins2 = placedb.net2pin_map[placedb.pin2net_map[pin]]
                    for pin2 in pins2:
                        isMacro1[placedb.pin2node_map[pin2]] = 1
        edge = []
        edge_weight = []
        Node_weight = Node_weight = [[] for _ in range(placedb.num_movable_nodes)]
        macro_weight=4
        for j in range(len(placedb.net_names)):
            if(len(placedb.net2pin_map[j])<2 or len(placedb.net2pin_map[j])> 50):
                continue
            else:
                weight = placedb.net_weights[j] / (len(placedb.net2pin_map[j])-1)
                for k in range(len(placedb.net2pin_map[j])-1):
                    num1 = placedb.net2pin_map[j][k]
                    Node1 = placedb.pin2node_map[num1]
                    for l in range(k+1,len(placedb.net2pin_map[j])):
                        num2 = placedb.net2pin_map[j][l]
                        Node2 = placedb.pin2node_map[num2]
                        if((Node1 == Node2)):
                            continue
                        else:
                            #macro와 연결된 node일 경우 edge weight를 4배 늘린다.
                            #Macro와 두다리 연결되어 있는 node의 경우 edge weight를 2배 늘린다.
                            if((Node2 < placedb.num_movable_nodes)):
                                if((Node1 < placedb.num_movable_nodes)):
                                    edge.append([Node1,Node2])
                                    if(isMacro[Node2]==1 or isMacro[Node1]==1):
                                        macro_weight = 4
                                    else:
                                        if(isMacro1[Node2]==1 or isMacro1[Node1]==1):
                                            macro_weight = 2
                                        else:
                                            macro_weight = 1
                                    edge_weight.append(weight*macro_weight)
                                    Node_weight[Node1].append([Node2,weight*macro_weight])
                                    Node_weight[Node2].append([Node1,weight*macro_weight])
                                else:
                                    if(isMacro[Node2]==1):
                                        macro_weight = 4
                                    else:
                                        macro_weight = 2 if isMacro1[Node2]==1 else 1
                                        Node_weight[Node2].append([Node1,weight*macro_weight])
                            else:
                                if((Node1 < placedb.num_movable_nodes)):
                                    if(isMacro[Node1]==1):
                                        macro_weight = 4
                                    else:
                                        macro_weight = 2 if isMacro1[Node1]==1 else 1
                                        Node_weight[Node1].append([Node2,weight*macro_weight])

        #content += "edge = %s\n" % (edge)
        #content += "edge_weight = %s\n" % (edge_weight)
        content += "macronum = %s\n" % (macronum)
        print(content)

        tt1 = time.time()
        logging.info("reading database takes %.2f seconds" % (tt1 - tt))
                     
        #graph 만들기
        g = ig.Graph(
            n= placedb.num_movable_nodes,
            edges = edge,
            edge_attrs={'weight' : edge_weight}
        )
        #print(g)
        #clustering 진행하기
        Clusteringresults= ig.Graph.community_leiden(g, objective_function="modularity", weights=edge_weight, resolution=1, beta=0.01, initial_membership=None, n_iterations=2, node_weights=None)
        tt2 = time.time()
        #print(Clusteringresults)
        logging.info("Clustering takes %.2f seconds" % (tt2 - tt1))
        
        #결과 출력하기
        """
        for i, community in enumerate(Clusteringresults):
            print(f"Community {i}:")
            for v in community:
                print(f"\t{v}")
        """
        #Cluster 수 중에서 갯수가 적은, Cluster안에서만 서로 연결되어 있는 Cluster는 없애기 - Cluster에 마지막 하나 추가해서 unplace된 애들 label 붙이기
        Clusters = []
        #Clusterarea = []
        Clusternum=0
        #totalclusterarea = 0
        Nodetocluster = [0] * placedb.num_movable_nodes
        rest = np.array([],dtype=np.int32)
        for i, community in enumerate(Clusteringresults):
            arr = np.array([],dtype=np.int32)
            #area = 0
            if(len(community))<10:
                for j in community:
                    rest = np.append(rest,j)
                continue
            else:
                for v in community:
                    arr = np.append(arr,v)
                    Nodetocluster[v]= Clusternum
                    #area += placedb.node_size_x[v] * placedb.node_size_y[v]
                Clusters.append(arr)
                #Clusterarea.append(area)
                #totalclusterarea += area
                Clusternum +=1

        for k in rest:
            Nodetocluster[k] = Clusternum

        #Cluster 결과 확인하기
        """
        for idx, array in enumerate(Clusters):
            print(f"Cluster{idx}: {array}")
            print(Clusterarea[idx])
        print(rest)
        print(Nodetocluster)
        print(Clusternum)
        """

        #Cluster 결과로 행렬 만들기 (A,Bx,By)
        A = np.zeros((len(Clusters), len(Clusters)),dtype=placedb.dtype)
        Bx = np.zeros((len(Clusters),1),dtype=placedb.dtype)
        By = np.zeros((len(Clusters),1),dtype=placedb.dtype)
        for i in range(len(Clusters)):
            for j in range(len(Clusters[i])):
                for k in range(len(Node_weight[Clusters[i][j]])):
                    if(Node_weight[Clusters[i][j]][k][0] >=placedb.num_movable_nodes):
                        Bx[i] += (Node_weight[Clusters[i][j]][k][1] * placedb.node_x[Node_weight[Clusters[i][j]][k][0]])
                        By[i] += (Node_weight[Clusters[i][j]][k][1] * placedb.node_y[Node_weight[Clusters[i][j]][k][0]])
                        A[i,i] += Node_weight[Clusters[i][j]][k][1]
                    else:
                        if(Nodetocluster[Node_weight[Clusters[i][j]][k][0]]<Clusternum and Nodetocluster[Node_weight[Clusters[i][j]][k][0]]!=i):
                            A[i,Nodetocluster[Node_weight[Clusters[i][j]][k][0]]] -=  Node_weight[Clusters[i][j]][k][1]
                            A[i,i] += Node_weight[Clusters[i][j]][k][1]
                        else:
                            if(Nodetocluster[Node_weight[Clusters[i][j]][k][0]]==Clusternum and Nodetocluster[Node_weight[Clusters[i][j]][k][0]]!=i):
                                Bx[i] += (Node_weight[Clusters[i][j]][k][1] * (placedb.xl * 1.0 + placedb.xh * 1.0) / 2)
                                By[i] += (Node_weight[Clusters[i][j]][k][1] * (placedb.yl * 1.0 + placedb.yh * 1.0) / 2)
                                A[i,i] += Node_weight[Clusters[i][j]][k][1]

        #Cluster 중에 외부와 연결이 안되어있는 경우 singular matrix가 되기 때문에 이를 방지하기 위해 빼주기 - Clusters에서도 빼줘야 한다.
        rows_with_zeros = np.all(A == 0, axis=1)
        A = np.delete(A, np.where(rows_with_zeros), axis=0)
        A = np.delete(A, np.where(rows_with_zeros), axis=1)
        Bx = np.delete(Bx, np.where(rows_with_zeros), axis=0)
        By = np.delete(By, np.where(rows_with_zeros), axis=0)
        indices = np.where(rows_with_zeros)[0]

        for i in reversed(indices):
            del Clusters[i]
            #totalclusterarea -= Clusterarea[i]
            #del Clusterarea[i]
            Clusternum -=1
    
        """
        for row in A:
            print(row)
        print(Bx)
        print(By)
        

        AverageClusterarea = totalclusterarea / Clusternum
        #Cluster size에 따라서 weight scaling 하기
        #1번째 함수
        
        for i in range(len(Clusters)-1):
            for j in range(i+1,len(Clusters)):
                if (Clusterarea[i] < AverageClusterarea * 0.1 and Clusterarea[j] < AverageClusterarea * 0.1):
                    A[i,j] = A[i,j] * 
                    A[j,i] = A[j,i] * 
                else:
                    if ((Clusterarea[i] < AverageClusterarea * 0.1 and Clusterarea[j] < totalclusterarea * 0.1) or (Clusterarea[j] < AverageClusterarea * 0.1 and Clusterarea[i] < totalclusterarea * 0.1)):
                        A[i,j] = A[i,j] * 
                        A[j,i] = A[j,i] * 
                if(Clusterarea[i] > totalclusterarea * 0.1 and Clusterarea[j]>totalclusterarea * 0.1):
                    A[i,j] = A[i,j] * 
                    A[j,i] = A[j,i] * 
                else :
                    if((Clusterarea[i] > totalclusterarea * 0.1 and Clusterarea[j]>AverageClusterarea * 0.1) or (Clusterarea[j] > totalclusterarea * 0.1 and Clusterarea[i]>AverageClusterarea * 0.1)):
                        A[i,j] = A[i,j] * 
                        A[j,i] = A[j,i] * 
        
        print(f"Cluster weight update")
        #2번째 함수
        for i in range(len(Clusters)-1):
            for j in range(i+1,len(Clusters)):
                if ((Clusterarea[i] < AverageClusterarea * 0.1 and Clusterarea[j] > totalclusterarea * 0.1) or (Clusterarea[j] < AverageClusterarea * 0.1 and Clusterarea[i] > totalclusterarea * 0.1)):
                    continue
                else :
                    if((Clusterarea[i] < totalclusterarea * 0.1 and Clusterarea[i]>AverageClusterarea * 0.1) and (Clusterarea[j] < totalclusterarea * 0.1 and Clusterarea[j]>AverageClusterarea * 0.1)):
                        continue
                    else:
                        value = AverageClusterarea * ((1/(Clusterarea[i]*Clusterarea[j]))**0.5)
                        weightupdate = round(value,3)
                        A[i,j] = A[i,j] * weightupdate
                        A[j,i] = A[j,i] * weightupdate
        """    
            
        #행렬 계산하기
        xx = np.linalg.solve(A,Bx)
        xy = np.linalg.solve(A,By)
        """
        print(xx)
        print(xy)
        """
        #계산 결과 바탕으로 cell location 정해주기
        initiallocation = np.zeros(placedb.num_movable_nodes * 2, dtype=placedb.dtype)
        initiallocation[0:placedb.num_movable_nodes] = np.random.normal(
                    loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
                    scale=(placedb.xh - placedb.xl) * 0.001,
                    size=placedb.num_movable_nodes)
        initiallocation[placedb.num_movable_nodes:placedb.num_movable_nodes*2]  =  np.random.normal(
                    loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
                    scale=(placedb.yh - placedb.yl) * 0.001,
                    size=placedb.num_movable_nodes)
        for i in range(len(Clusters)):
            for j in range(len(Clusters[i])):
                initiallocation[Clusters[i][j]] = xx[i]
                initiallocation[placedb.num_movable_nodes+Clusters[i][j]] = xy[i]
        """
        print(initiallocation)
        """
        tt3 = time.time()
        logging.info("Initial location takes %.2f seconds" % (tt3 - tt2)) 
        logging.info("Initial Placements takes %.2f seconds" % (tt3 - tt1)) 
        return  initiallocation