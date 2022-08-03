package com.microsoft.z3

fun Model.translate(target: Context): Model {
    val translated = Native.modelTranslate(context.nCtx(), nativeObject, target.nCtx())
    return Model(target, translated)
}
