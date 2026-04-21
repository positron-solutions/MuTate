// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// XXX update docs for VUID-01796, writes must include all stages that can see the bytes written to
// (no partial updates)
// XXX update docs for VUID-00292, no two ranges may share a stage flag

#![allow(unused)]

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
    Data, Fields, Token,
};

// DEBT Scalar block.  Hardcoded DataLayouts.  How would the user switch to 430 easily?  Suppose we
// could indirect defaults through an alias and then feature-gate the alias.
// XXX Recognize stage shorthand identifiers.  Several non-ambiguous constants and abbreviations are
// completely readable and the user can't complete names like

/// The common interface for both direct push constant declaration and inline declaration inside
/// pipeline declarations.  Either parsed content of a `push!(Name { field: Type, ... })`
/// invocation, or equivalently the data extracted from a `#[derive(Push)]` DeriveInput.
pub(crate) struct PushAttr {
    pub name: syn::Ident,
    pub fields: Vec<syn::Field>,
}

impl Parse for PushAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: syn::Ident = input.parse()?;
        let inner;
        syn::braced!(inner in input);
        let fields = Punctuated::<syn::Field, Token![,]>::parse_terminated_with(&inner, |p| {
            syn::Field::parse_named(p)
        })?
        .into_iter()
        .collect();
        Ok(PushAttr { name, fields })
    }
}

/// Track spans to pinpoint issues after checks.
#[derive(Clone)]
struct VisibleAttr {
    stages: Vec<syn::Ident>,
    span: Span,
}

/// Parse `#[visible(...)]`, a `|`-separated list of identifiers.
fn parse_visible(attrs: &[syn::Attribute]) -> syn::Result<Option<VisibleAttr>> {
    for attr in attrs {
        if !attr.path().is_ident("visible") {
            continue;
        }
        let span = attr.span();
        // Parse
        let ts: TokenStream = attr.parse_args()?;
        let mut stages = Vec::new();
        for tt in ts {
            match tt {
                proc_macro2::TokenTree::Ident(id) => stages.push(id),
                proc_macro2::TokenTree::Punct(p) if p.as_char() == '|' => {}
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "expected a shader stage identifier or `|`",
                    ))
                }
            }
        }
        if stages.is_empty() {
            return Err(syn::Error::new(
                span,
                "#[visible(...)] requires at least one stage",
            ));
        }
        return Ok(Some(VisibleAttr { stages, span }));
    }
    Ok(None)
}

/// Everything we need to know about one struct field for range-building.
struct FieldInfo<'f> {
    /// Original syn field, kept for span-accurate error reporting.
    field: &'f syn::Field,
    /// `None` ⇒ ALL stages.
    visible: Option<VisibleAttr>,
}

/// Tracks the first and last field index (in declaration order) visible to one stage.
struct StageCoverage {
    /// The stage identifier as written, `RayGen`, `Closest`.
    stage: syn::Ident,
    first: usize,
    last: usize,
}

struct MergedRange {
    /// OR of all stage flags that share this exact span.
    stages: Vec<syn::Ident>,
    first: usize,
    last: usize,
}

/// Returns one entry per distinct stage name, in first-appearance order.
fn collect_stage_coverage(fields: &[FieldInfo<'_>]) -> Vec<StageCoverage> {
    let mut coverage: Vec<StageCoverage> = Vec::new();
    // Index of the most recent all-stages field, if any.
    let mut all_floor: Option<usize> = None;

    for (idx, fi) in fields.iter().enumerate() {
        let stage_names: &[syn::Ident] = match &fi.visible {
            None => &[],
            Some(v) => &v.stages,
        };

        if stage_names.is_empty() {
            // All-stages field: extend every existing entry's last...
            for cov in &mut coverage {
                cov.last = idx;
            }
            // ...and record the floor so future explicit stages start here.
            all_floor = Some(idx);
        } else {
            for stage_id in stage_names {
                let key = stage_id.to_string();
                match coverage.iter_mut().find(|c| c.stage.to_string() == key) {
                    Some(cov) => cov.last = idx,
                    None => coverage.push(StageCoverage {
                        stage: stage_id.clone(),
                        // Floor: if an all-stages field appeared before this
                        // stage's first explicit mention, it must be included.
                        first: all_floor.map_or(idx, |f| f.min(idx)),
                        last: idx,
                    }),
                }
            }
        }
    }

    coverage
}

fn merge_ranges(coverage: Vec<StageCoverage>) -> Vec<MergedRange> {
    let mut merged: Vec<MergedRange> = Vec::new();
    for cov in coverage {
        match merged
            .iter_mut()
            .find(|m| m.first == cov.first && m.last == cov.last)
        {
            Some(m) => m.stages.push(cov.stage),
            None => merged.push(MergedRange {
                stages: vec![cov.stage],
                first: cov.first,
                last: cov.last,
            }),
        }
    }
    merged
}

fn emit_range(struct_name: &syn::Ident, range: &MergedRange) -> TokenStream {
    let bits_expr =
        range
            .stages
            .iter()
            .fold(quote! { ::ash::vk::ShaderStageFlags::empty() }, |acc, s| {
                quote! {
                    ::ash::vk::ShaderStageFlags::from_raw(
                        #acc.as_raw()
                            | <::mutate_vulkan::pipeline::stage::#s
                               as ::mutate_vulkan::pipeline::stage::StageSlot>::FLAGS.as_raw()
                    )
                }
            });

    let first = range.first;
    let last = range.last;

    let gpu_ty = quote! {
        <#struct_name as ::mutate_vulkan::slang::GpuType<::mutate_vulkan::slang::Scalar>>
    };

    let offset_expr = quote! {
        ::mutate_vulkan::slang::field_start(
            &#gpu_ty::FIELD_NODE,
            #first,
            ::mutate_vulkan::slang::Scalar::DATA_LAYOUT,
        ) as u32
    };
    let size_expr = quote! {
        (::mutate_vulkan::slang::field_end(
            &#gpu_ty::FIELD_NODE,
            #last,
            ::mutate_vulkan::slang::Scalar::DATA_LAYOUT,
        ) - ::mutate_vulkan::slang::field_start(
            &#gpu_ty::FIELD_NODE,
            #first,
            ::mutate_vulkan::slang::Scalar::DATA_LAYOUT,
        )) as u32
    };

    quote! {
        ::ash::vk::PushConstantRange {
            stage_flags: #bits_expr,
            offset:      #offset_expr,
            size:        #size_expr,
        }
    }
}

/// Emit the struct definition and all trait impls for a push constants type.
///
/// Called from both [`derive_push`] (where the struct already exists in user
/// code and must **not** be re-emitted) and [`emit_push_inline`] (where the
/// struct is being synthesised from a `push!(Name { ... })` invocation inside
/// a pipeline macro and must be emitted).
///
/// `emit_struct` controls whether the `struct` definition itself is part of
/// the output — the derive path sets it to `false`; the inline path sets it
/// to `true`.
fn emit_push_impls(
    name: &syn::Ident,
    fields: &[syn::Field],
    emit_struct: bool,
) -> syn::Result<TokenStream> {
    let field_infos: syn::Result<Vec<FieldInfo<'_>>> = fields
        .iter()
        .map(|f| {
            let visible = parse_visible(&f.attrs)?;
            Ok(FieldInfo { field: f, visible })
        })
        .collect();
    let field_infos = field_infos?;

    let all_unannotated = field_infos.iter().all(|f| f.visible.is_none());
    let all_annotated = field_infos.iter().all(|f| f.visible.is_some());

    if !all_unannotated && !all_annotated {
        let first_bare = field_infos
            .iter()
            .find(|f| f.visible.is_none())
            .map(|f| f.field.span())
            .unwrap();
        return Err(syn::Error::new(
            first_bare,
            "mixed #[visible] annotations: either annotate every field \
             (disjoint per-stage ranges) or annotate none (one ALL range). \
             Unannotated fields in a mixed struct would require ALL, which \
             cannot coexist with any other range per VUID-00292.",
        ));
    }

    let gpu_ty = quote! {
        <#name as ::mutate_vulkan::slang::GpuType<::mutate_vulkan::slang::Scalar>>
    };

    let ranges: Vec<TokenStream> = if field_infos.is_empty() {
        vec![] // zero-range layout — perfectly legal, nothing to compute
    } else if all_unannotated {
        let n = field_infos.len();
        vec![quote! {
            ::ash::vk::PushConstantRange {
                stage_flags: ::ash::vk::ShaderStageFlags::ALL,
                offset: 0,
                size: ::mutate_vulkan::slang::field_end(
                    &#gpu_ty::FIELD_NODE,
                    #n - 1,
                    ::mutate_vulkan::slang::Scalar::DATA_LAYOUT,
                ) as u32,
            }
        }]
    } else {
        let coverage = collect_stage_coverage(&field_infos);
        let merged = merge_ranges(coverage);

        let mut seen: Vec<String> = Vec::new();
        for m in &merged {
            for stage in &m.stages {
                let key = stage.to_string();
                if seen.contains(&key) {
                    return Err(syn::Error::new(
                        stage.span(),
                        format!(
                            "stage `{key}` appears in more than one push constant range \
                             (VUID-00292): each stage flag bit may be set in at most one range"
                        ),
                    ));
                }
                seen.push(key);
            }
        }

        merged.iter().map(|r| emit_range(name, r)).collect()
    };

    // XXX the default DataLayout, Scalar, might have drifted around when it should not.  No idea
    // how we would specify a different layout here, but attempting to would probably shove the
    // problem into the bright sunlight.
    // Optionally emit the struct itself (inline path only).
    let struct_def = if emit_struct {
        let struct_fields: Vec<syn::Field> = fields
            .iter()
            .map(|f| {
                let mut f = f.clone();
                f.attrs.retain(|a| !a.path().is_ident("visible"));
                f
            })
            .collect();

        quote! {
            #[derive(::mutate_macros::GpuType)]
            #[repr(C)]
            pub struct #name {
                #(#struct_fields),*
            }
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #struct_def

        impl ::mutate_vulkan::pipeline::push::PushConstants for #name {}

        impl ::mutate_vulkan::pipeline::layout::LayoutSpec for #name {
            type D    = ::mutate_vulkan::slang::Scalar;
            type Push = Self;
            const RANGES: &'static [::ash::vk::PushConstantRange] = &[ #(#ranges),* ];
        }
    })
}

/// Called by `#[derive(Push)]`.  The struct already exists; emit impls only.
pub(crate) fn derive_push(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    crate::force::assert_repr_c(&input.attrs, input.ident.span())?;

    let named_fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "#[derive(Push)] requires a struct with named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "#[derive(Push)] can only be applied to structs",
            ))
        }
    };

    let fields: Vec<syn::Field> = named_fields.iter().cloned().collect();
    emit_push_impls(&input.ident, &fields, /* emit_struct = */ false)
}

/// Called directly from pipeline macros that have already parsed a `PushAttr`.
/// Emits the struct definition plus all trait impls without a re-parse round-trip.
pub(crate) fn emit_push_from_attr(attr: PushAttr) -> syn::Result<TokenStream> {
    emit_push_impls(&attr.name, &attr.fields, /* emit_struct = */ true)
}

/// Called by `push!(Name { ... })` at the top level.
pub(crate) fn emit_push_inline(input: TokenStream) -> syn::Result<TokenStream> {
    let attr = syn::parse2::<PushAttr>(input)?;
    emit_push_from_attr(attr)
}
