# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
common function for build
"""

from te.platform.cce_build import build_config_update
from te.platform.cce_build import build_config


def set_bool_storage_config():
    """
    update build config
    set is_bool_storage_as_1bit as false
    :return:
    """
    config = build_config_update(build_config, "bool_storage_as_1bit", False)
    return build_config_update(config, "double_buffer_non_reuse", True)
