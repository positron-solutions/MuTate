// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// XXX update docs for VUID-01796, writes must include all stages that can see the bytes written to
// (no partial updates)
// XXX update docs for VUID-00292, no two ranges may share a stage flag

#![allow(unused)]

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{spanned::Spanned, Data, Fields};

// DEBT Scalar block.  Hardcoded DataLayouts.  How would the user switch to 430 easily?  Suppose we
// could indirect defaults through an alias and then feature-gate the alias.
// XXX Recognize stage shorthand identifiers.  Several non-ambiguous constants and abbreviations are
// completely readable and the user can't complete names like

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
    /// The stage identifier as written, e.g. `RAYGEN`, `CLOSEST`.
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
    let bits_expr = range.stages.iter().fold(
        quote! { 0u32 },
        |acc, s| quote! { #acc | ::ash::vk::ShaderStageFlags::#s.as_raw() },
    );
    let stage_flags = quote! {
        ::ash::vk::ShaderStageFlags::from_raw(#bits_expr)
    };

    let first = range.first;
    let last = range.last;

    let gpu_ty = quote! {
        <#struct_name as ::mutate_vulkan::slang::GpuType<::mutate_vulkan::slang::Scalar>>
    };

    // offset = byte start of the first field in the range
    // size   = byte end of the last field minus that offset
    // Both are evaluated entirely at compile time via const fn.
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
            stage_flags: #stage_flags,
            offset:      #offset_expr,
            size:        #size_expr,
        }
    }
}

pub(crate) fn derive_push(input: &syn::DeriveInput) -> syn::Result<TokenStream> {
    let vis = &input.vis;

    // Named-field struct only.
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

    //
    crate::force::assert_repr_c(&input.attrs, input.ident.span())?;

    let name = &input.ident;
    let layout_name = quote::format_ident!("{name}Layout");

    // Walk fields and map field stage flags into ranges.  Legal range declaration for pipelines
    // requires that no two ranges share the same flag at the same byte!
    // XXX verify behavior
    let field_infos: syn::Result<Vec<FieldInfo<'_>>> = named_fields
        .iter()
        .map(|f| {
            let visible = parse_visible(&f.attrs)?;
            Ok(FieldInfo { field: f, visible })
        })
        .collect();
    let field_infos = field_infos?;

    let coverage = collect_stage_coverage(&field_infos);
    let merged = merge_ranges(coverage);
    let ranges: Vec<TokenStream> = merged.iter().map(|range| emit_range(name, range)).collect();

    // XXX the default DataLayout, Scalar, might have drifted around when it should not.  No idea
    // how we would specify a different layout here, but attempting to would probably shove the
    // problem into the bright sunlight.
    Ok(quote! {
        impl ::mutate_vulkan::pipeline::push::PushConstants for #name {}

        impl ::mutate_vulkan::pipeline::layout::LayoutSpec for #name
        {
            type D    = ::mutate_vulkan::slang::Scalar;
            type Push = Self;
            const RANGES: &'static [::ash::vk::PushConstantRange] = &[ #(#ranges),* ];
        }
    })
}
