# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

# Currently Paying Down

None.  We are still identifying high interest rates.

# Charging Interest

Each element includes two parts:

- A description of the problem being managed and how it may be solved better later.
- "For now" instructions to minimize the cost of interest that will be paid when cleaning up the debt.

## Error Handling

Probably using thiserror to facilitate shipping as a library.

### For Now

- Return Result types from fallible operations to ensure proper combinator usages
- Unwrap and panic liberally (but do **not** clone haphazardly!)

## Vulkan Versions & Device Compatibility

Anticipate monolithic platform builds that switch at runtime for more specific support.

### For Now

- Use 1.3+ and any extensions from 1.4 that enhance productivity significantly
- Use `cfg` gates only for platforms, not for Vulkan versions.  To switch on Vulkan support, use runtime conditions.
