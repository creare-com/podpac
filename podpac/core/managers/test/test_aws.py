import pytest
import os


class TestAWS(object):
    def test_old_module_deprecation(self):
        with pytest.warns(DeprecationWarning):
            import podpac.core.managers.aws_lambda

        assert podpac.core.managers.aws_lambda.Lambda
