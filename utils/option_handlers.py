from click import Option, UsageError

class RequiredByWhenSetTo(Option):
    def __init__(self, *args, **kwargs):
        self.required_by = kwargs.pop("required_by")
        self.set_to = kwargs.pop("set_to")
        assert self.required_by, "'required_by' parameter required"
        assert self.set_to, "'set_to' parameter required"
        kwargs["help"] = (kwargs.get("help", "") +
            f" NOTE: This argument is required when {self.required_by} is set to {self.set_to}"
        ).strip()
        super(RequiredByWhenSetTo, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        self_present = self.name in opts
        required_by_set_correct = opts[self.required_by] == self.set_to

        if required_by_set_correct:
            if not self_present:
                raise UsageError(
                    f"Illegal usage: {self.name} is required when {self.required_by} is set to {self.set_to}"
                )

        return super(RequiredByWhenSetTo, self).handle_parse_result(ctx, opts, args)
