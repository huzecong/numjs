// Type definitions for numjs 0.14
// Project: https://github.com/nicolaspanel/numjs#readme
// Definitions by: taoqf <https://github.com/taoqf>
//                 matt <https://github.com/mattmm3d>
// Definitions: https://github.com/DefinitelyTyped/DefinitelyTyped
// TypeScript Version: 2.3

export as namespace nj;
import * as BaseNdArray from 'ndarray';

export type NdType<T> = BaseNdArray.DataType | BaseNdArray.Data<T>;

type Slice = null | {
    0: number | null;
    1: number | null;
    2?: number | null;
};

export interface NdArray<T = number> extends BaseNdArray<T> {
    readonly size: number;
    readonly shape: number[];
    readonly ndim: number;
    dtype: BaseNdArray.DataType;
    readonly T: NdArray<T>;

    get(...args: number[]): T;

    set(...args: number[]): T;

    slice(...args: Array<number | Slice>): NdArray<T>;

    /**
     * Return a subarray by fixing a particular axis
     * @example
     *   arr = nj.arange(4*4).reshape(4,4)
     *   // array([[  0,  1,  2,  3],
     *   //        [  4,  5,  6,  7],
     *   //        [  8,  9, 10, 11],
     *   //        [ 12, 13, 14, 15]])
     *
     *   arr.pick(1)
     *   // array([ 4, 5, 6, 7])
     *
     *   arr.pick(null, 1)
     *   // array([  1,  5,  9, 13])
     */
    pick(...axis: number[]): NdArray<T>

    /**
     * Return a shifted view of the array. Think of it as taking the upper left corner of the image and dragging it inward
     * @example
     *   arr = nj.arange(4*4).reshape(4,4)
     *   // array([[  0,  1,  2,  3],
     *   //        [  4,  5,  6,  7],
     *   //        [  8,  9, 10, 11],
     *   //        [ 12, 13, 14, 15]])
     *   arr.lo(1,1)
     *   // array([[  5,  6,  7],
     *   //        [  9, 10, 11],
     *   //        [ 13, 14, 15]])
     */
    lo(...args: number[]): NdArray<T>

    /**
     * Return a sliced view of the array.
     * @example
     *   arr = nj.arange(4*4).reshape(4,4)
     *   // array([[  0,  1,  2,  3],
     *   //        [  4,  5,  6,  7],
     *   //        [  8,  9, 10, 11],
     *   //        [ 12, 13, 14, 15]])
     *
     *   arr.hi(3,3)
     *   // array([[  0,  1,  2],
     *   //        [  4,  5,  6],
     *   //        [  8,  9, 10]])
     *
     *   arr.lo(1,1).hi(2,2)
     *   // array([[ 5,  6],
     *   //        [ 9, 10]])
     */
    hi(...args: number[]): NdArray<T>

    /**
     * Changes the stride length by rescaling. Negative indices flip axes.
     * @example  Create a reversed view of a 1D array.
     *   const reversed = a.step(-1)
     * @example  Split an array into even and odd components.
     *   const evens = a.step(2)
     *   const odds = a.lo(1).step(2)
     */
    step(...args: number[]): NdArray<T>;

    /**
     * Return a copy of the array collapsed into one dimension using row-major order (C-style)
     */
    flatten<P>(): NdArray<P>;

    /**
     * Permute the dimensions of the array.
     */
    transpose(axes?: number[]): NdArray<T>;

    transpose(...axis: number[]): NdArray<T>;

    /**
     * Dot product of two arrays.
     */
    dot(x: NjArray<T>): NdArray<T>;

    /**
     * Assign `x` to the array, element-wise.
     */
    assign(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Add `x` to the array, element-wise.
     */
    add(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Subtract `x` to the array, element-wise.
     */
    subtract(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Multiply array by `x`, element-wise.
     */
    multiply(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Divide array by `x`, element-wise.
     */
    divide(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Raise array elements to powers from given array, element-wise.
     *
     * @param [copy=true] - set to false to modify the array rather than create a new one
     */
    pow(x: NjParam<T>, copy?: boolean): NdArray<T>;

    /**
     * Calculate the exponential of all elements in the array, element-wise.
     *
     * @param [copy=true] - set to false to modify the array rather than create a new one
     */
    exp(copy?: boolean): NdArray<T>;

    /**
     * Calculate the natural logarithm of all elements in the array, element-wise.
     *
     * @param {boolean} [copy=true] - set to false to modify the array rather than create a new one
     * @returns {NdArray}
     */
    log(copy?: boolean): NdArray<T>;

    /**
     * Calculate the positive square-root of all elements in the array, element-wise.
     *
     * @param [copy=true] - set to false to modify the array rather than create a new one
     */
    sqrt(copy?: boolean): NdArray<T>;

    /**
     * Return the maximum value of the array
     */
    max(): T;

    /**
     * Return the minimum value of the array
     */
    min(): T;

    /**
     * Sum of array elements.
     */
    sum(): T;

    /**
     * Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
     */
    std(): number;

    /**
     * Return the arithmetic mean of array elements.
     */
    mean(): T;

    /**
     * Converts {NdArray} to a native JavaScript {Array}
     */
    tolist<LT = T>(): LT[];

    valueOf<LT = T>(): LT[];

    /**
     * Stringify the array to make it readable in the console, by a human.
     */
    toString(): string;

    inspect(): string;

    /**
     * Stringify object to JSON
     */
    toJSON(): any;

    /**
     * Create a full copy of the array
     */
    clone(): NdArray<T>;

    /**
     * Return true if two arrays have the same shape and elements, false otherwise.
     */
    equal<U>(array: NjArray<U>): boolean;

    /**
     * Round array to the to the nearest integer.
     */
    round(copy?: boolean): NdArray<T>;

    /**
     * Return the inverse of the array, element-wise.
     */
    negative(): NdArray<T>;

    diag(): NdArray<T>;

    iteraxis(axis: number, cb: (x: NdArray<T>, i: number) => any): void;

    /**
     * Returns the discrete, linear convolution of the array using the given filter.
     *
     * @note: Arrays must have the same dimensions and `filter` must be smaller than the array.
     * @note: The convolution product is only given for points where the signals overlap completely. Values outside the signal boundary have no effect. This behaviour is known as the 'valid' mode.
     * @note: Use optimized code for 3x3, 3x3x1, 5x5, 5x5x1 filters, FFT otherwise.
     */
    convolve(filter: NjArray<T>): NdArray<T>;

    fftconvolve(filter: NjArray<T>): NdArray<T>;
}

type RecursiveArray<T> = T[] | RecursiveArray<T>[];
export type NdArrayData<T> = BaseNdArray.Data<T>;
export type NjArray<T> = NdArrayData<T> | NdArray<T> | RecursiveArray<T>;
export type NjParam<T> = NjArray<T> | number;

/**
 * Return absolute value of the input array, element-wise.
 *
 */
export function abs<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Add arguments, element-wise.
 *
 */
export function add<T = number>(a: NjParam<T>, b: NjParam<T>): NdArray<T>;

export function arange<T = number>(stop: number, dtype?: NdType<T>): NdArray<T>;
export function arange<T = number>(start: number, stop: number, dtype?: NdType<T>): NdArray<T>;
/**
 * Return evenly spaced values within a given interval.
 *
 * @param [start = 0] Start of interval. The interval includes this value.
 * @param stop End of interval. The interval does not include this value.
 * @param [step = 1] Spacing between values. The default step size is 1. If step is specified, start must also be given.
 * @param [dtype = Array] The type of the output array.
 * @returns Array of evenly spaced values.
 */
export function arange<T = number>(start: number, stop: number, step: number, dtype?: NdType<T>): NdArray<T>;

/**
 * Return trigonometric inverse cosine of the input array, element-wise.
 *
 */
export function arccos<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Return trigonometric inverse sine of the input array, element-wise.
 *
 */
export function arcsin<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Return trigonometric inverse tangent of the input array, element-wise.
 *
 */
export function arctan<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Clip (limit) the values in an array between min and max, element-wise.
 * @param [min = 0]
 * @param [max = 1]
 */
export function clip<T = number>(x: NjParam<T>, min?: number, max?: number): NdArray<T>;

/**
 * Join given arrays along the last axis.
 *
 */
export function concatenate<T = number>(...arrays: Array<NjArray<T>>): NdArray<T>;

/**
 * Convolve 2 N-dimensionnal arrays
 *
 */
export function convolve<T = number>(a: NjArray<T>, b: NjArray<T>): NdArray<T>;

/**
 * Return trigonometric cosine of the input array, element-wise.
 *
 */
export function cos<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Divide `a` by `b`, element-wise.
 *
 */
export function divide<T = number>(a: NjArray<T>, b: NjParam<T>): NdArray<T>;

/**
 * Dot product of two arrays. WARNING: supported products are: - matrix dot matrix - vector dot vector - matrix dot vector - vector dot matrix
 *
 */
export function dot<T = number>(a: NjArray<T>, b: NjArray<T>): NdArray<T>;

/**
 * Return a new array of given shape and type, filled with `undefined` values.
 *
 * @param shape    Shape of the new array, e.g., [2, 3] or 2.
 * @param [dtype]    The type of the output array.
 * @returns Array of `undefined` values with the given shape and dtype
 */
export function empty<T = number>(shape: NdArrayData<T> | number, dtype?: NdType<T>): NdArray<T>;

/**
 * Return true if two arrays have the same shape and elements, false otherwise.
 *
 */
export function equal<T = number>(a: NjArray<T>, b: NjArray<T>): boolean;

/**
 * Calculate the exponential of all elements in the input array, element-wise.
 *
 */
export function exp<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Convolve 2 N-dimensionnal arrays using Fast Fourier Transform (FFT)
 *
 */
export function fftconvolve<T = number>(a: NjArray<T>, b: NjArray<T>): NdArray<T>;

/**
 * Return a copy of the array collapsed into one dimension using row-major order (C-style)
 *
 */
export function flatten<T = number>(array: NjArray<T>): NdArray<T>;

/**
 * Reverse the order of elements in an array along the given axis.
 * The shape of the array is preserved, but the elements are reordered.
 * New in version 0.15.0.
 *
 * @param axis  Axis in array, which entries are reversed.
 * @return A view of `m` with the entries of axis reversed.  Since a view is returned, this operation is done in constant time.
 */
export function flip<T = number>(array: NjArray<T>, axis: number): NdArray<T>;

export function getRawData<T = number>(array: NdArrayData<T>): Uint8Array;

export function setRawData<T = number>(array: NdArrayData<T>, data: NdArrayData<T>): Uint8Array;

/**
 * Compute the leaky-ReLU value for each array element.
 *
 * @param [alpha = 1e-3]
 */
export function leakyRelu<T = number>(array: NjArray<T>, alpha: number): NdArray<T>;

/**
 * Calculate the natural logarithm of all elements in the input array, element-wise.
 */
export function log<T = number>(array: NjArray<T>): NdArray<T>;


/**
 * Return the maximum value of the array
 *
 */
export function max<T = number>(x: NjParam<T>): T;

/**
 * Return the arithmetic mean of input array elements.
 *
 */
export function mean<T = number>(x: NjParam<T>): T;

/**
 * Return the minimum value of the array
 *
 */
export function min<T = number>(x: NjParam<T>): T;

/**
 * Return element-wise remainder of division.
 * Computes the remainder complementary to the `floor` function. It is equivalent to the Javascript modulus operator``x1 % x2`` and has the same sign as the divisor x2.
 */
export function mod<T = number>(x1: NjParam<T>, x2: NjParam<T>): NdArray<T>;

/**
 * Multiply arguments, element-wise.
 *
 */
export function multiply<T = number>(a: NjArray<T>, b: NjParam<T>): NdArray<T>;

/**
 * Return the inverse of the input array, element-wise.
 *
 */
export function negative<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Return a new array of given shape and type, filled with ones.
 *
 * @param shape  Shape of the new array, e.g., [2, 3] or 2.
 * @param [dtype]  The type of the output array.
 * @returns Array of ones with the given shape and dtype
 */
export function ones<T = number>(shape: NdArrayData<T> | number, dtype?: BaseNdArray.DataType): NdArray<T>;

/**
 * Raise first array elements to powers from second array, element-wise.
 *
 */
export function power<T = number>(x1: NjParam<T>, x2: NjParam<T>): NdArray<T>;

/**
 * Create an array of the given shape and propagate it with random samples from a uniform distribution over [0, 1].
 *
 * @param [shape]  The dimensions of the returned array, should all be positive integers
 */
export function random<T = number>(shape?: NdArrayData<T> | number): NdArray<T>;

/**
 * Return element-wise remainder of division.
 * Computes the remainder complementary to the `floor` function. It is equivalent to the Javascript modulus operator``x1 % x2`` and has the same sign as the divisor x2.
 */
export function remainder<T = number>(x1: NjParam<T>, x2: NjParam<T>): NdArray<T>;

/**
 * Gives a new shape to an array without changing its data.
 *
 * @param shape The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length
 */
export function reshape<T = number>(array: NjArray<T>, shape: NdArray<T>): NdArray<T>;

/**
 * Rotate an array by 90 degrees in the plane specified by axes.
 * Rotation direction is from the first towards the second axis.
 * New in version 0.15.0.
 *
 * @param [k = 1]  Number of times the array is rotated by 90 degrees.
 * @param [axes = [0,1]]  The array is rotated in the plane defined by the axes. Axes must be different.
 * @return A rotated view of m.
 */
export function rot90<T = number>(array: NjArray<T>, k?: number, axes?: number[]): NdArray<T>;

/**
 * Round an array to the to the nearest integer.
 *
 */
export function round<T = number>(x: NjArray<T>): NdArray<T>;

/**
 * Return the sigmoid of the input array, element-wise.
 *
 * @param [t = 1]  stiffness parameter
 */
export function sigmoid<T = number>(x: NjParam<T>, t?: number): NdArray<T>;

/**
 * Return trigonometric sine of the input array, element-wise.
 *
 */
export function sin<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Return the softmax, or normalized exponential, of the input array, element-wise.
 *
 */
export function softmax<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Calculate the positive square-root of all elements in the input array, element-wise.
 *
 */
export function sqrt<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Returns the standard deviation, a measure of the spread of a distribution, of the input array elements.
 *
 */
export function std<T = number>(x: NjParam<T>): T;

/**
 * Subtract second argument from the first, element-wise.
 *
 */
export function subtract<T = number>(a: NjParam<T>, b: NjParam<T>): T;

/**
 * Return the sum of input array elements.
 *
 */
export function sum<T = number>(x: NjParam<T>): T;

/**
 * Return trigonometric tangent of the input array, element-wise.
 *
 */
export function tan<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Return hyperbolic tangent of the input array, element-wise.
 *
 */
export function tanh<T = number>(x: NjParam<T>): NdArray<T>;

/**
 * Permute the dimensions of the input array according to the given axes.
 *
 * @example
 *
 * arr = nj.arange(6).reshape(1,2,3)
 * // array([[[ 0, 1, 2],
 * //         [ 3, 4, 5]]])
 * arr.T
 * // array([[[ 0],
 * //         [ 3]],
 * //        [[ 1],
 * //         [ 4]],
 * //        [[ 2],
 * //         [ 5]]])
 * arr.transpose(1,0,2)
 * // array([[[ 0, 1, 2]],
 * //        [[ 3, 4, 5]]])
 */
export function transpose<T = number>(x: NjParam<T>, axes?: number): NdArray<T>;

/**
 * Return a new array of given shape and type, filled with zeros.
 *
 * @param shape Shape of the new array, e.g., [2, 3] or 2.
 * @param [dtype = Array] The type of the output array.
 * @returns Array of zeros with the given shape and dtype
 */
export function zeros<T = number>(shape: NdArrayData<T> | number, dtype?: BaseNdArray.DataType): NdArray<T>;

export namespace errors {
    function ValueError(message?: string): Error;

    function ConfigError(message?: string): Error;

    function NotImplementedError(message?: string): Error;
}

export function broadcast<T, U>(shape1: T[], shape2: U[]): Array<T | U>;

export function fft<T = number>(x: NjArray<T>): NdArray<T>;

export function ifft<T = number>(x: NjArray<T>): NdArray<T>;

/**
 * Extract a diagonal or construct a diagonal array.
 *
 * @returns a view a of the original array when possible, a new array otherwise
 */
export function diag<T = number>(x: NjArray<T>): NdArray<T>;

/**
 * The identity array is a square array with ones on the main diagonal.
 * @param Number of rows (and columns) in n x n output.
 * @param  [dtype=Array]  The type of the output array.
 * @return n x n array with its main diagonal set to one, and all other elements 0
 */
export function identity<T = number>(n: T, dtype?: BaseNdArray.DataType): NdArray<T>;

/**
 * Join a sequence of arrays along a new axis.
 * The axis parameter specifies the index of the new axis in the dimensions of the result.
 * For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
 * @param sequence of array_like
 * @param [axis=0] The axis in the result array along which the input arrays are stacked.
 * @return The stacked array has one more dimension than the input arrays.
 */
export function stack<T = number>(arrays: Array<NdArray<T>>, axis?: number): NdArray<T>;

export function array<T = number>(arr: NjArray<T>, dtype?: BaseNdArray.DataType): NdArray<T>;

export function int8<T = number>(arr: NjArray<T>): NjArray<Int8Array>;

export function uint8<T = number>(arr: NjArray<T>): NjArray<Uint8Array>;

export function int16<T = number>(arr: NjArray<T>): NjArray<Int16Array>;

export function uint16<T = number>(arr: NjArray<T>): NjArray<Uint16Array>;

export function int32<T = number>(arr: NjArray<T>): NjArray<Int32Array>;

export function uint32<T = number>(arr: NjArray<T>): NjArray<Uint32Array>;

export function float32<T = number>(arr: NjArray<T>): NjArray<Float32Array>;

export function float64<T = number>(arr: NjArray<T>): NjArray<Float64Array>;

export namespace config {
    export let printThreshold: number;
    export let nFloatingValues: number;
}

export namespace dtypes {
    export const int8: Int8ArrayConstructor;
    export const int16: Int16ArrayConstructor;
    export const int32: Int32ArrayConstructor;
    export const uint8: Uint8ArrayConstructor;
    export const uint16: Uint16ArrayConstructor;
    export const uint32: Uint32ArrayConstructor;
    export const float32: Float32ArrayConstructor;
    export const float64: Float64ArrayConstructor;
    export const array: ArrayConstructor;
}
