# Schema repair, v1

Your previous tool call did not satisfy the tool's input schema.

Validation error:

${error}

Return a single call to the same tool whose input satisfies the schema exactly.
Do not change your substantive conclusions to make validation pass; correct
only the shape of the output. If a required field genuinely cannot be
determined from the input, say so in that field's text rather than inventing a
value for it.
