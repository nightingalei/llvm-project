//===- SPIRVCooperativeTensorOps.td - cooperative operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of cooperative tensor extension ops.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_COOPERATIVE_TENSOR_OPS
#define SPIRV_COOPERATIVE_TENSOR_OPS

// -----

def SPV_CooperativeTensorLengthVSIOp : SPV_Op<"CooperativeTensorLengthVSI",
  [NoSideEffect]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Number of components of a cooperative tensor type accessible to each
    invocation when treated as a composite.

    Result Type must be an OpTypeInt with 32-bit Width and 0 Signedness.

    Type is a cooperative tensor type.

    ``` {.ebnf}
    cooperative-tensor-length-op ::= ssa-id `=` `spv.CooperativeTensorLengthVSI
                                    ` : ` cooperative-tensor-type
    ```

    For example:

    ```
    %0 = spv.CooperativeTensorLengthVSI : !spv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = "attr-dict `:` $type";

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    TypeAttr:$type
  );

  let results = (outs
    SPV_Int32:$result
  );
  let verifier = [{ return success(); }];
}

// -----

def SPV_CooperativeTensorLoadVSIOp : SPV_Op<"CooperativeTensorLoadVSI", []> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Load a cooperative tensor through a pointer.

    Result Type is the type of the loaded object. It must be a cooperative
    tensor type.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    For a given dynamic instance of this instruction, all operands of this
    instruction must be the same for all invocations in a given scope instance
    (where the scope is the scope the cooperative tensor type was created with).
    All invocations in a given scope instance must be active or all must be
    inactive.

    ### Custom assembly form

    ``` {.ebnf}
    cooperative-tensorload-op ::= ssa-id `=` `spv.CooperativeTensorLoadVSI`
                              ssa-use `,` ssa-use `,` ssa-use
                              (`[` memory-access `]`)? ` : `
                              pointer-type `as`
                              cooperative-tensor-type
    ```

    For example:

    ```
    %0 = spv.CooperativeTensorLoadVSI %ptr, %offset, %strides, %hwcLayout
         : !spv.ptr<i32, StorageBuffer> as !spv.cooptensor<16x8xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_Integer:$offset,
    SPV_Vector:$strides,
    SPV_Bool:$hwclayout,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs
    SPV_AnyCooperativeTensor:$result
  );

  let verifier = [{
    return verifyPointerAndCoopTensorType(*this, pointer().getType(),
                                          result().getType());
  }];
}

// -----

def SPV_CooperativeTensorMatMulAddVSIOp : SPV_Op<"CooperativeTensorMatMulAddVSI",
  [NoSideEffect, AllTypesMatch<["c", "result"]>]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Linear-algebraic tensor multiply of A by B and then component-wise add C.
    The order of the operations is implementation-dependent. The internal
    precision of floating-point operations is defined by the client API.
    Integer operations are performed at the precision of the Result Type and are
    exact unless there is overflow or underflow, in which case the result is
    undefined.

    Result Type must be a cooperative tensor type with M rows and N columns.

    A is a cooperative tensor with M rows and K columns.

    B is a cooperative tensor with K rows and N columns.

    C is a cooperative tensor with M rows and N columns.

    The values of M, N, and K must be consistent across the result and operands.
    This is referred to as an MxNxK tensor multiply.

    A, B, C, and Result Type must have the same scope, and this defines the
    scope of the operation. A, B, C, and Result Type need not necessarily have
    the same component type, this is defined by the client API.

    If the Component Type of any tensor operand is an integer type, then its
    components are treated as signed if its Component Type has Signedness of 1
    and are treated as unsigned otherwise.

    For a given dynamic instance of this instruction, all invocations in a given
    scope instance must be active or all must be inactive (where the scope is
    the scope of the operation).

    ``` {.ebnf}
    cooperative-tensormatmuladd-op ::= ssa-id `=` `spv.CooperativeTensorMatMulAddVSI`
                              ssa-use `,` ssa-use `,` ssa-use ` : `
                              a-cooperative-tensor-type,
                              b-cooperative-tensor-type ->
                              result-cooperative-tensor-type
    ```
    For example:

    ```
    %0 = spv.CooperativeTensorMatMulAddVSI %arg0, %arg1, %arg2,  :
      !spv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($a) `,` type($b) `->` type($c)
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPV_AnyCooperativeTensor:$a,
    SPV_AnyCooperativeTensor:$b,
    SPV_AnyCooperativeTensor:$c
  );

  let results = (outs
    SPV_AnyCooperativeTensor:$result
  );

  let verifier = [{ return verifyCoopTensorMatMulAdd(*this); }];
}

// -----

def SPV_CooperativeTensorStoreVSIOp : SPV_Op<"CooperativeTensorStoreVSI", []> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    Store a cooperative tensor through a pointer.

    Pointer is a pointer into an array. Its type must be an OpTypePointer whose
    Type operand is a scalar or vector type. The storage class of Pointer must
    be Workgroup, StorageBuffer, or (if SPV_EXT_physical_storage_buffer is
    supported) PhysicalStorageBufferEXT.

    Object is the object to store. Its type must be an
    OpTypeCooperativeTensorVSI.

    Memory Access must be a Memory Access literal. If not present, it is the
    same as specifying None.

    ``` {.ebnf}
    coop-tensor-store-op ::= `spv.CooperativeTensorStoreVSI `
                              ssa-use `, ` ssa-use `, `
                              ssa-use `, ` ssa-use `, `
                              (`[` memory-access `]`)? `:`
                              pointer-type `,` spirv-element-type
    ```

    For example:

    ```
      spv.CooperativeTensorStoreVSI %arg0, %arg2, %arg1, %arg3 :
        !spv.ptr<i32, StorageBuffer>, !spv.cooptensor<16x8xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_AnyCooperativeTensor:$object,
    SPV_Integer:$offset,
    SPV_Vector:$strides,
    SPV_Bool:$hwclayout,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access
  );

  let results = (outs);

  let verifier = [{
    return verifyPointerAndCoopTensorType(*this, pointer().getType(),
                                          object().getType());
  }];
}

// -----

def SPV_CooperativeTensorConv2DVSIOp : SPV_Op<"CooperativeTensorConv2DVSI",
  [NoSideEffect]> {
  let summary = "See extension SPV_VSI_cooperative_tensor";

  let description = [{
    ``` {.ebnf}
    cooperative-tensormatmuladd-op ::= ssa-id `=` `spv.CooperativeTensorConv2DVSI`
                              ssa-use `,` ssa-use `,` ssa-use `,` ssa-use ` : `
                              input-cooperative-tensor-type,
                              weight-cooperative-tensor-type,
                              bias-cooperative-tensor-type,
                              strides-vector-type ->
                              result-cooperative-tensor-type
    ```
    For example:

    ```
    %0 = spv.CooperativeTensorConv2DVSI %input, %weight, %bias, %strides  :
      !spv.cooptensor<i32, 8, 16>
    ```
  }];

  let assemblyFormat = [{
    operands attr-dict`:` type($input) `,` type($weight) `,` type($bias) `,` type($strides) `->` type($result)
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[SPV_VSI_cooperative_tensor]>,
    Capability<[SPV_C_CooperativeTensorVSI]>
  ];

  let arguments = (ins
    SPV_AnyCooperativeTensor:$input,
    SPV_AnyCooperativeTensor:$weight,
    SPV_AnyCooperativeTensor:$bias,
    SPV_Vector:$strides
  );

  let results = (outs
    SPV_AnyCooperativeTensor:$result
  );

  let verifier = [{ return verifyCoopTensorConv2D(*this); }];
}

// -----

#endif // SPIRV_COOPERATIVE_TENSOR_OPS
