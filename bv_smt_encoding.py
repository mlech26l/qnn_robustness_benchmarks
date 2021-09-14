from __future__ import division, print_function
import numpy as np
import pyboolector
from pyboolector import Boolector, BoolectorException
import quantization_util as qu
import time


def _renormalize(product, excessive_bits):
    shift_bits = excessive_bits
    residue = product % (2 ** shift_bits)
    c = product // (2 ** shift_bits)
    if residue >= (2 ** (shift_bits - 1)):
        c += 1

    return np.int32(c)



def propagate_dense(in_layer, out_layer, w, b):
    for out_index in range(out_layer.layer_size):
        weight_row = w[:, out_index]
        bias_factor = (
            in_layer.frac_bits
            + in_layer.quantization_config["int_bits_bias"]
            - in_layer.quantization_config["int_bits_weights"]
        )

        bias = np.int32(b[out_index] * (2 ** bias_factor))

        bound_1 = weight_row * in_layer.clipped_lb
        bound_2 = weight_row * in_layer.clipped_ub
        # Need unclipped bounds to allocate minimal amount of bits
        # TODO: Problem: intermediate result may explode? Does it matter?
        accumulator_lb = np.minimum(bound_1, bound_2).sum() + bias
        accumulator_ub = np.maximum(bound_1, bound_2).sum() + bias

        accumulator_bits = (
            in_layer.bit_width + in_layer.quantization_config["quantization_bits"]
        )
        accumulator_frac = in_layer.frac_bits + (
            in_layer.quantization_config["quantization_bits"]
            - in_layer.quantization_config["int_bits_weights"]
        )

        excessive_bits = accumulator_frac - (
            in_layer.quantization_config["quantization_bits"]
            - in_layer.quantization_config["int_bits_activation"]
        )

        lb = _renormalize(accumulator_lb, excessive_bits)
        ub = _renormalize(accumulator_ub, excessive_bits)
        # No activation function in the output layer in general
        if out_layer.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )


        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        out_layer.accumulator_lb[out_index] = accumulator_lb
        out_layer.accumulator_ub[out_index] = accumulator_ub

        out_layer.clipped_lb[out_index] = clipped_lb
        out_layer.clipped_ub[out_index] = clipped_ub

        out_layer.lb[out_index] = lb
        out_layer.ub[out_index] = ub



class LayerEncoding:
    def __init__(
        self,
        layer_size,
        btor,
        bit_width,
        frac_bits,
        quantization_config,
        signed_output=False,
    ):
        self.layer_size = layer_size
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.quantization_config = quantization_config
        self.signed_output = signed_output

        self.vars = [
            btor.Var(btor.BitVecSort(self.bit_width)) for i in range(layer_size)
        ]

        if self.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        self.clipped_lb = min_val * np.ones(layer_size, dtype=np.int32)
        self.clipped_ub = max_val * np.ones(layer_size, dtype=np.int32)

        acc_min, acc_max = qu.int_get_min_max_integer(30, None)
        self.accumulator_lb = acc_min * np.ones(layer_size, dtype=np.int32)
        self.accumulator_ub = acc_max * np.ones(layer_size, dtype=np.int32)

        self.lb = acc_min * np.ones(layer_size, dtype=np.int32)
        self.ub = acc_max * np.ones(layer_size, dtype=np.int32)

    def set_bounds(self, low, high, is_input_layer=False):
        self.lb = low
        self.ub = high

        if is_input_layer:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["input_bits"],
                self.quantization_config["input_bits"]
                - self.quantization_config["int_bits_input"],
            )
        elif self.signed_output:
            # No activation function in the output layer in general
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )


        self.clipped_lb = np.clip(self.lb, min_val, max_val)
        self.clipped_ub = np.clip(self.ub, min_val, max_val)


class QNNEncoding:
    def __init__(self, quantized_model, btor=None, verbose=None, config=None):

        if btor is None:
            btor = Boolector()
            btor.Set_opt(pyboolector.BTOR_OPT_MODEL_GEN, 2)
        self.btor = btor
        self._debug_list = []
        self._verbose = verbose
        # This setting should yield the best results on average
        self.config = {
            "add_bound_constraints": True,
            "sat_engine_cadical": True,
            "rebalance_sum": True,
            "recursive_sum": True,
            "sort_sum": None,
            "propagate_bounds": True,
            "relu_simplify": True,
            "subsum_elimination": False, # subsub elimination can hurt performance -> better not activate it
            "min_bits": True,
            "shift_elimination": True,
        }
        # Overwrite default config with argument
        if not config is None:
            for k, v in config.items():
                self.config[k] = v

        self._stats = {
            "constant_neurons": 0,
            "linear_neurons": 0,
            "unstable_neurons": 0,
            "reused_expressions": 0,
            "partially_stable_neurons": 0,
            "build_time": 0,
            "sat_time": 0,
            "total_time": 0,
        }

        self.dense_layers = []
        self.quantized_model = quantized_model

        self._last_layer_signed = quantized_model._last_layer_signed
        self.quantization_config = quantized_model.quantization_config

        current_bits = self.quantization_config["input_bits"]
        for i, l in enumerate(quantized_model.dense_layers):
            self.dense_layers.append(
                LayerEncoding(
                    layer_size=l.units,
                    btor=self.btor,
                    bit_width=self.quantization_config["quantization_bits"],
                    frac_bits=self.quantization_config["quantization_bits"]
                    - self.quantization_config["int_bits_activation"],
                    quantization_config=self.quantization_config,
                    signed_output=l.signed_output,
                )
            )
        # Create input vars
        input_size = quantized_model._input_shape[-1]
        self.input_layer = LayerEncoding(
            layer_size=input_size,
            btor=self.btor,
            bit_width=self.quantization_config["input_bits"],
            frac_bits=self.quantization_config["input_bits"]
            - self.quantization_config["int_bits_input"],
            quantization_config=self.quantization_config,
        )
        self.input_vars = self.input_layer.vars
        self.output_vars = self.dense_layers[-1].vars

    def print_verbose(self, text):
        if self._verbose:
            print(text)

    def only_encode(self):
        if self.config["propagate_bounds"]:
            self.propagate_bounds()
        self.encode()

    def sat(self, timeout=None, verbose=False):
        build_start_time = time.time()
        if self.config["propagate_bounds"]:
            if verbose:
                print("Propagating bounds ...")
            self.propagate_bounds()

        if verbose:
            print("Encode model ...")
        self.encode()

        if not timeout is None:
            self.btor.Set_term(lambda x: time.time() - x > timeout, time.time())

        if self.config["sat_engine_cadical"]:
            self.btor.Set_sat_solver("cadical")
        else:
            self.btor.Set_sat_solver("lingeling")

        if verbose:
            print("Invoking SMT engine ...")
        sat_start_time = time.time()
        result = self.btor.Sat()
        end_time = time.time()
        smt_total_seconds = end_time - build_start_time
        self._stats["build_time"] = sat_start_time - build_start_time
        self._stats["sat_time"] = end_time - sat_start_time
        self._stats["total_time"] = end_time - build_start_time

        if verbose:
            smt_mins = smt_total_seconds // 60
            smt_hours = int(smt_mins // 60)
            smt_mins = int(smt_mins % 60)
            smt_seconds = int(smt_total_seconds % 60)
            print(
                "SMT runtime {:02d}:{:02d}:{:02d} h:m:s".format(
                    smt_hours, smt_mins, smt_seconds
                )
            )

        if result != self.btor.SAT and result != self.btor.UNSAT:
            raise ValueError(
                "Ooops '{}' returned by SMT engine, this should never happen!".format(
                    str(result)
                )
            )

        return result == self.btor.SAT

    def propagate_bounds(self):

        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            self.print_verbose("propagate Dense layer")
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            propagate_dense(current_layer, l, w, b)

            current_layer = l

    def encode(self):

        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            self.print_verbose("encoding Dense layer")
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            self.encode_dense(current_layer, l, w, b)

            current_layer = l

    def get_accumulator_bits_layerwide(self, in_layer, out_layer, b):
        return 1 + int(
            np.max(
                [
                    self.get_accumulator_bits_individually(in_layer, out_layer, b, i)
                    for i in range(out_layer.layer_size)
                ]
            )
        )

    def get_accumulator_bits_individually(self, in_layer, out_layer, b, out_index):
        bias_factor = (
            in_layer.frac_bits
            + self.quantization_config["int_bits_bias"]
            - self.quantization_config["int_bits_weights"]
        )
        bias = int(b[out_index] * (2 ** bias_factor))

        abs_max_value = 1 + np.max(
            [
                np.abs(out_layer.accumulator_lb[out_index]),
                np.abs(out_layer.accumulator_ub[out_index]),
            ]
        )
        # print("abs_max_value: ",str(abs_max_value))
        # print("bias: ",str(bias))
        accumulator_bits = np.max(
            [
                int(np.ceil(np.log2(abs_max_value))) + 1,
                int(np.ceil(np.log2(np.abs(bias) + 1))) + 1,
                8,
            ]
        )
        return 1 + accumulator_bits

    def encode_dense(self, in_layer, out_layer, w, b):

        self.clear_scratchpad()
        if self.config["subsum_elimination"]:
            if self.config["min_bits"]:
                accumulator_bits = 1 + self.get_accumulator_bits_layerwide(
                    in_layer, out_layer, b
                )
            else:
                accumulator_bits = 32
            sign_extended_x = [
                self.btor.Uext(in_layer.vars[i], accumulator_bits - in_layer.bit_width)
                for i in range(in_layer.layer_size)
            ]
            self.subexpression_analysis(in_layer, sign_extended_x, w)
        for out_index in range(out_layer.layer_size):
            weight_row = w[:, out_index]
            bias_factor = (
                in_layer.frac_bits
                + self.quantization_config["int_bits_bias"]
                - self.quantization_config["int_bits_weights"]
            )
            bias = int(b[out_index] * (2 ** bias_factor))

            if not self.config["subsum_elimination"]:
                # In subsum mode we sign extend all vars to max bit_size of layer
                if self.config["min_bits"]:
                    accumulator_bits = 1 + self.get_accumulator_bits_individually(
                        in_layer, out_layer, b, out_index
                    )
                else:
                    accumulator_bits = 32
                sign_extended_x = [
                    self.btor.Uext(
                        in_layer.vars[i], accumulator_bits - in_layer.bit_width
                    )
                    for i in range(in_layer.layer_size)
                ]
            if not self.config["min_bits"]:
                accumulator_bits = 32

            id_var_weight_list = [
                (i, sign_extended_x[i], int(weight_row[i]))
                for i in range(len(sign_extended_x))
            ]
            if self.config["rebalance_sum"]:
                id_var_weight_list = self.prune_zeros(id_var_weight_list)
            if not self.config["sort_sum"] is None:
                id_var_weight_list = self.sort_sum(
                    id_var_weight_list,
                    ascending=self.config["sort_sum"].startswith("asc"),
                )
            accumulator = self.reduce_MAC(
                id_var_weight_list, subsum_elimination=self.config["subsum_elimination"]
            )

            # We only care about reachable values of the accumulator. Overflows do not affect the final value
            # accumulator_overflow = 2**accumulator_bits
            # bias = int(bias%accumulator_overflow)

            if accumulator is None:
                # Neuron is constant 0
                accumulator = self.btor.Const(bias, accumulator_bits)
            else:
                accumulator = self.btor.Add(accumulator, bias)

            accumulator_bits = (
                in_layer.bit_width + self.quantization_config["quantization_bits"]
            )
            accumulator_frac = in_layer.frac_bits + (
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_weights"]
            )

            excessive_bits = accumulator_frac - (
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"]
            )

            self.renormalize(
                value=accumulator,
                lb=out_layer.lb[out_index],
                ub=out_layer.ub[out_index],
                output_var=out_layer.vars[out_index],
                signed_output=out_layer.signed_output,
                excessive_bits=excessive_bits,
                aux="dense {} out_index: {}".format(out_layer.signed_output, out_index),
            )

    def prune_zeros(self, id_var_weight_list):
        new_list = []
        for i, x, w in id_var_weight_list:
            if w != 0:
                new_list.append((i, x, w))
        return new_list

    def sort_sum(self, id_var_weight_list, ascending=True):

        factor = 1 if ascending else -1
        id_var_weight_list.sort(key=lambda tup: factor * int(np.abs(tup[-1])))

        return id_var_weight_list

    def clear_scratchpad(self):
        self._scatchpad = {}

    def _add_expr_to_scratchpad(self, in_index, w_value, expr):
        if not in_index in self._scatchpad.keys():
            self._scatchpad[in_index] = {}
        self._scatchpad[in_index][w_value] = expr

    def _get_expr_from_scratchpad(self, in_index, w_value):
        try:
            return self._scatchpad[in_index][w_value]
        except:
            s = "None"
            if in_index in self._scatchpad.keys():
                s = str(self._scatchpad[in_index])
            print("\nQuery: [{}][{}]".format(in_index, w_value))
            print(
                "Screthpad key error! keys: {}, item: {}".format(
                    self._scatchpad.keys(), s
                )
            )
            print("ERROR")
            import sys

            sys.exit(-1)

    def _build_pos_expression(self, in_index, v, all_pos_values):
        sext_in_var = self._get_expr_from_scratchpad(in_index, 1)
        expr = None
        assert not sext_in_var is None
        if v == 1:
            return sext_in_var

        if self.config["shift_elimination"]:
            for shift in [1, 2, 3, 4, 5, 6]:
                mul_val = 2 ** shift
                if (
                    v > mul_val
                    and all_pos_values[v // mul_val] > 0
                    and v % mul_val == 0
                ):
                    # Add one extra bit to avoid overflow
                    expr = self._get_expr_from_scratchpad(in_index, v // mul_val)
                    expr = self.btor.Mul(expr, mul_val)
                    return expr

        # No heuristics found to reuse some values for this expression
        expr = self.btor.Mul(sext_in_var, v)
        return expr

    def _build_neg_expression(self, in_index, v, all_pos_values, all_neg_values):
        if all_pos_values[v] > 0:
            # print("Reuse found by negate")
            sext_in_var = self._get_expr_from_scratchpad(in_index, v)
            assert not sext_in_var is None
            return self.btor.Neg(sext_in_var)

        if self.config["shift_elimination"]:
            for shift in [1, 2, 3, 4, 5, 6]:
                mul_val = 2 ** shift
                if (
                    v > mul_val
                    and all_neg_values[v // mul_val] > 0
                    and v % mul_val == 0
                ):
                    # Add one extra bit to avoid overflow
                    expr = self._get_expr_from_scratchpad(in_index, -v // mul_val)
                    expr = self.btor.Mul(expr, mul_val)
                    return expr

        # No heuristics found to reuse some values for this expression
        sext_in_var = self._get_expr_from_scratchpad(in_index, 1)
        assert not sext_in_var is None
        expr = self.btor.Mul(sext_in_var, -v)
        return expr

    def subexpression_analysis(self, in_layer, sext_in_vars, weight_matrix):
        for in_index in range(weight_matrix.shape[0]):
            all_pos_values = np.zeros(
                2 ** (self.quantization_config["quantization_bits"] - 1) + 1,
                dtype=np.int32,
            )
            all_neg_values = np.zeros(
                2 ** (self.quantization_config["quantization_bits"] - 1) + 1,
                dtype=np.int32,
            )
            for o in range(weight_matrix.shape[1]):
                w = int(weight_matrix[in_index, o])
                if w >= 0:
                    all_pos_values[w] += 1
                else:
                    all_neg_values[-w] += 1
            # Force generation of 1
            self._add_expr_to_scratchpad(in_index, 1, sext_in_vars[in_index])
            for v in range(
                1, 2 ** (self.quantization_config["quantization_bits"] - 1) + 1
            ):
                if all_pos_values[v] > 0:
                    expr = self._build_pos_expression(in_index, v, all_pos_values)
                    self._add_expr_to_scratchpad(in_index, v, expr)
                if all_neg_values[v] > 0:
                    expr = self._build_neg_expression(
                        in_index, v, all_pos_values, all_neg_values
                    )
                    self._add_expr_to_scratchpad(in_index, -v, expr)

    def reduce_MAC(self, id_var_weight_list, subsum_elimination):
        if len(id_var_weight_list) == 1:
            i, x, weight_value = id_var_weight_list[0]
            if weight_value == 0:
                return None
            if subsum_elimination:
                expr = self._get_expr_from_scratchpad(i, weight_value)
                if expr is None:
                    raise ValueError(
                        "Missing expression x[{}]*{}".format(i, weight_value)
                    )
                if x.width - expr.width > 0:
                    expr = self.btor.Sext(expr, x.width - expr.width)
                    return expr
                    print(
                        "WARNING: Sub-sum has more bits than total sum ({} and {}) falling back to standard multiplication".format(
                            str(x.width), str(expr.width)
                        )
                    )

            if weight_value == 1:
                return x
            elif weight_value == -1:
                return self.btor.Neg(x)
            else:
                return self.btor.Mul(x, weight_value)
        else:
            if self.config["recursive_sum"]:
                center = len(id_var_weight_list) // 2
            else:
                center = 1
            left = self.reduce_MAC(id_var_weight_list[:center], subsum_elimination)
            right = self.reduce_MAC(id_var_weight_list[center:], subsum_elimination)
            if left is None:
                return right
            elif right is None:
                return left
            return self.btor.Add(left, right)

    def renormalize(
        self, value, lb, ub, output_var, signed_output, excessive_bits, aux=None
    ):
        assert lb <= ub
        if signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        # Prune constraints, if we already know the value of the neuron
        if self.config["relu_simplify"] and clipped_ub == clipped_lb:
            self.btor.Assert(self.btor.Eq(output_var, int(clipped_ub)))
            self.print_verbose("neuron fixed at {}".format(int(clipped_ub)))
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_lb >= max_val:
            self.print_verbose("neuron fixed at ub {}".format(max_val))
            self.btor.Assert(self.btor.Eq(output_var, max_val))
            self._stats["constant_neurons"] += 1
            return
        elif self.config["relu_simplify"] and clipped_ub <= min_val:
            self.btor.Assert(self.btor.Eq(output_var, min_val))
            self.print_verbose("neuron fixed at lb {}".format(min_val))
            self._stats["constant_neurons"] += 1
            return

        # bit_slice = input_bits-1

        residue = self.btor.Slice(value, excessive_bits - 1, 0)
        quotient = self.btor.Slice(
            value, value.width - 1, excessive_bits
        )  # of size quotient bits

        # Quotient (+1) depending on residue
        rouned_output = self.btor.Var(
            self.btor.BitVecSort(value.width - excessive_bits)
        )

        residue_threshold = 2 ** (excessive_bits - 1)
        self.btor.Assert(
            (
                self.btor.Ugte(residue, residue_threshold)
                & self.btor.Eq(rouned_output, self.btor.Inc(quotient))
            )
            | (
                self.btor.Not(self.btor.Ugte(residue, residue_threshold))
                & self.btor.Eq(rouned_output, quotient)
            )
        )

        # Output var
        # with same number of bits as rounded_ouput
        sign_ext_func = self.btor.Uext
        gte_func = self.btor.Ugte
        lte_func = self.btor.Ulte
        gt_func = self.btor.Ugt
        lt_func = self.btor.Ult
        if signed_output:
            gte_func = self.btor.Sgte
            lte_func = self.btor.Slte
            gt_func = self.btor.Sgt
            lt_func = self.btor.Slt
            sign_ext_func = self.btor.Sext
        if rouned_output.width - output_var.width < 0:
            rouned_output = self.btor.Sext(
                rouned_output, -(rouned_output.width - output_var.width)
            )
        output_var_sign_ext = sign_ext_func(
            output_var, rouned_output.width - output_var.width
        )

        if (self.config["relu_simplify"]) and lb >= min_val and ub <= max_val:
            # no clipping possible, "linear" neuron
            self.btor.Assert(self.btor.Eq(output_var_sign_ext, rouned_output))
            self.print_verbose(
                "neuron is linear: [{}, {}] clippling range: [{}, {}]".format(
                    lb, ub, min_val, max_val
                )
            )
            self._stats["linear_neurons"] += 1
        elif self.config["relu_simplify"] and lb >= min_val:
            # low clipping is impossible
            self.btor.Assert(
                (
                    self.btor.Sgt(rouned_output, max_val)
                    & self.btor.Eq(output_var, max_val)
                )
                | (
                    self.btor.Slte(rouned_output, max_val)
                    & self.btor.Eq(rouned_output, output_var_sign_ext)
                )
            )
            self.print_verbose("neuron cannot reach min_val of {}".format(min_val))
            self._stats["partially_stable_neurons"] += 1
        elif self.config["relu_simplify"] and ub <= max_val:
            # high clipping is impossible
            self.btor.Assert(
                (
                    self.btor.Slt(rouned_output, min_val)
                    & self.btor.Eq(output_var, min_val)
                )
                | (
                    self.btor.Sgte(rouned_output, min_val)
                    & self.btor.Eq(rouned_output, output_var_sign_ext)
                )
            )
            self.print_verbose("neuron cannot reach max val of {}".format(max_val))
            self._stats["partially_stable_neurons"] += 1
        else:
            # Don't know anything about the neuron
            self.btor.Assert(
                (
                    self.btor.Slt(rouned_output, min_val)
                    & self.btor.Eq(output_var, min_val)
                )
                | (
                    self.btor.Sgt(rouned_output, max_val)
                    & self.btor.Eq(output_var, max_val)
                )
                | (
                    self.btor.Sgte(rouned_output, min_val)
                    & self.btor.Slte(rouned_output, max_val)
                    & self.btor.Eq(output_var_sign_ext, rouned_output)
                )
            )
            # self.print_verbose("dont't know anything about neuron")
            self._stats["unstable_neurons"] += 1

        # Add bound constraints, output_var (4 bit) is unsigned!!!!
        if self.config["add_bound_constraints"]:
            if clipped_lb > min_val:
                self.btor.Assert(gte_func(output_var, int(clipped_lb)))
            if clipped_ub < max_val:
                self.btor.Assert(lte_func(output_var, int(clipped_ub)))
            # self._debug_list.append((output_var,int(clipped_lb),int(clipped_ub),aux))

    def print_stats(self):
        print("Constant neurons: {:d}".format(self._stats["constant_neurons"]))
        print("Linear   neurons: {:d}".format(self._stats["linear_neurons"]))
        print("Bistable neurons: {:d}".format(self._stats["partially_stable_neurons"]))
        print(
            "Unstable neurons: {:d} (tri-stable)".format(self._stats["unstable_neuron"])
        )

    # Assumes low and high are already quantized
    def assert_input_box(self, low, high, set_bounds=True):
        input_size = len(self.input_vars)

        # Ensure low is a vector
        low = np.array(low, dtype=np.int32) * np.ones(input_size, dtype=np.int32)
        high = np.array(high, dtype=np.int32) * np.ones(input_size, dtype=np.int32)

        saturation_min, saturation_max = qu.uint_get_min_max_integer(
            self.quantization_config["input_bits"],
            self.quantization_config["input_bits"]
            - self.quantization_config["int_bits_input"],
        )
        low = np.clip(low, saturation_min, saturation_max)
        high = np.clip(high, saturation_min, saturation_max)
        # Make sure variables bounds can be represented
        if set_bounds:
            self.input_layer.set_bounds(low, high, is_input_layer=True)

        for i in range(input_size):
            self.btor.Assert(self.btor.Ugte(self.input_layer.vars[i], int(low[i])))
            self.btor.Assert(self.btor.Ulte(self.input_layer.vars[i], int(high[i])))

    def sanitize(self):
        for var, lb, ub, aux in self._debug_list:
            value = qu.binary_str_to_uint(var.assignment)
            if value < lb or value > ub:
                print(
                    "ERROR: value: {}, range [{}, {}] aux: {}".format(
                        value, lb, ub, str(aux)
                    )
                )

    def assert_not_argmax(self, max_index):
        lte_func = self.btor.Slte if self._last_layer_signed else self.btor.Ulte
        lt_func = self.btor.Slt if self._last_layer_signed else self.btor.Ult

        disjunction = None
        for i in range(len(self.output_vars)):
            if i == int(max_index):
                continue
            # Ensure that maximum (max decision) is NOT max_index
            if i < int(max_index):
                or_term = lte_func(
                    self.output_vars[int(max_index)], self.output_vars[i]
                )
            else:
                or_term = lt_func(self.output_vars[int(max_index)], self.output_vars[i])
            if disjunction is None:
                disjunction = or_term
            else:
                disjunction = self.btor.Or(disjunction, or_term)

        self.btor.Assert(disjunction)

    def assert_argmax(self, var_list, max_index):
        gte_func = self.btor.Sgte if self._last_layer_signed else self.btor.Ugte
        gt_func = self.btor.Sgt if self._last_layer_signed else self.btor.Ugt

        for i in range(len(var_list)):
            if i == int(max_index):
                continue
            # Ensure that maximum (max decision) IS target_output_index
            if i < int(max_index):
                # Argmax returns the first maximum, therefore we must
                # not allow equality of outputs that come first
                self.btor.Assert(gte_func(var_list[int(max_index)], var_list[i]))
            else:
                # Equality is possible for outputs that come after max_index
                self.btor.Assert(gt_func(var_list[int(max_index)], var_list[i]))
                # print("Assert out[{:d}] >= out[{:d}]".format(int(target_output_index),i))

    def assert_input_distance(self, var_list, distance):
        for i in range(len(var_list)):

            greater = self.btor.Ugte(self.input_vars[i], var_list[i])
            self.btor.Assert(
                self.btor.Implies(
                    greater,
                    self.btor.Ulte(
                        self.btor.Sub(self.input_vars[i], var_list[i]), distance
                    ),
                )
            )
            self.btor.Assert(
                self.btor.Implies(
                    self.btor.Not(greater),
                    self.btor.Ulte(
                        self.btor.Sub(var_list[i], self.input_vars[i]), distance
                    ),
                )
            )

    def assert_not_output_distance(self, var_list, distance):
        self._output_dbg_expr = []
        for i in range(len(var_list)):
            if self._last_layer_signed:
                # Add one extra bit to make unsigned into signed
                v1 = self.btor.Sext(self.output_vars[i], 1)
                v2 = self.btor.Sext(var_list[i], 1)
            else:
                # not needed if output is already signed
                v1 = self.btor.Uext(self.output_vars[i], 1)
                v2 = self.btor.Uext(var_list[i], 1)
            sub = self.btor.Sub(v1, v2)
            expr = self.btor.Sgte(sub, int(distance)) | self.btor.Slte(
                sub, -int(distance)
            )
            self.btor.Assert(expr)
            self._output_dbg_expr.append(expr)

    def get_input_assignment(self):
        input_values = np.array(
            [qu.binary_str_to_uint(var.assignment) for var in self.input_layer.vars]
        )
        return np.array(input_values, dtype=np.int32)

    def get_forward_buffer(self):
        print("_last_layer_signed: ", str(self._last_layer_signed))
        input_values = np.array(
            [qu.binary_str_to_uint(var.assignment) for var in self.input_layer.vars],
            dtype=np.int32,
        )
        # activation_buffer = [input_values]
        activation_buffer = []
        for i in range(len(self.dense_layers) - 1):
            values = np.array(
                [
                    qu.binary_str_to_uint(var.assignment)
                    for var in self.dense_layers[i].vars
                ],
                dtype=np.int32,
            )
            activation_buffer.append(values)
        decode_func = (
            qu.binary_str_to_int if self._last_layer_signed else qu.binary_str_to_uint
        )
        output_values = np.array(
            [decode_func(var.assignment) for var in self.output_vars], dtype=np.int32
        )
        activation_buffer.append(output_values)
        return activation_buffer

    def get_output_assignment(self):
        output_values = np.empty(len(self.output_vars), dtype=np.int32)
        decode_func = (
            qu.binary_str_to_int if self._last_layer_signed else qu.binary_str_to_uint
        )
        for i in range(len(self.output_vars)):
            output_values[i] = decode_func(self.output_vars[i].assignment)

        return qu.de_quantize_uint(
            output_values,
            self.quantization_config["quantization_bits"],
            self.quantization_config["quantization_bits"]
            - self.quantization_config["int_bits_activation"],
        )


def check_robustness(encoding, x, y, eps):
    """
        Return values: 
            np.array: Adversarial attack if one exists
            True:     If sample is robust
            None:     If timeout or other error occured
    """
    x = x.flatten()
    low, high = x - eps, x + eps
    encoding.assert_input_box(low, high)
    encoding.assert_not_argmax(int(y))
    attack = None
    if encoding.sat(verbose=False):
        attack = encoding.get_input_assignment()
        return (False, attack)
    else:
        return (True, None)

def export_robustness(filename,encoding, x, y, eps):
    x = x.flatten()
    low, high = x - eps, x + eps
    encoding.assert_input_box(low, high)
    encoding.assert_not_argmax(int(y))
    encoding.only_encode()
    encoding.btor.Dump(format="smt2",outfile=filename)