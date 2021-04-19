#!/bin/bash

# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_VERSION_INFO_FILE="${_CURR_PATH}""/../../version.info"
_FILELIST_FILE="${_CURR_PATH}""/../../filelist.csv"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"

_INSTALL_LOG_DIR="/opp/install_log"
_INSTALL_INFO_SUFFIX="/opp/ascend_install.info"
_VERSION_INFO_SUFFIX="/opp/version.info"

PARAM_INVALID="0x0002"
INSTALL_FAILED="0x0000"
INSTALL_FAILED_DES="Update successed."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
FILE_WRITE_FAILED="0x0081"
FILE_WRITE_FAILED_DES="File write failed."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

# log functions
function getDate() {
    local _cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "${_cur_date}"
}

function logAndPrint() {
    local is_error_level=$(echo $1 | grep -E 'ERR_NO|WARN|INFO')
    if [[ "${is_quiet}" != "y" ]] || [[ "${is_error_level}" != "" ]]; then
        echo "[opp] ""$1"
    fi
    echo "[opp] [$(getDate)] ""$1" >> "${_INSTALL_LOG_FILE}"
}

function logWithErrorLevel() {
    local _ret_status="$1"
    local _level="$2"
    local _msg="$3"
    if [[ "${_ret_status}" != 0 ]]; then
        if [[ "${_level}" == "error" ]]; then
            logAndPrint "${_msg}"
            exit 1
        else
            logAndPrint "${_msg}"
        fi
    fi
}

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="UserName"
KEY_INSTALLED_UGROUP="UserGroup"
KEY_INSTALLED_TYPE="Opp_Install_Type"
KEY_INSTALLED_PATH="Opp_Install_path_Param"
KEY_INSTALLED_PATH_BUGFIX="Opp_Install_Path_Param"
KEY_INSTALLED_VERSION="Opp_Version"
function getInstalledInfo() {
    local _key="$1"
    local _res=""
    if [[ -f "${_INSTALL_INFO_FILE}" ]]; then
        chmod 644 "${_INSTALL_INFO_FILE}" 2> /dev/null
        . "${_INSTALL_INFO_FILE}"
        case "${_key}" in
        UserName)
            res=$(echo ${UserName})
            ;;
        UserGroup)
            res=$(echo ${UserGroup})
            ;;
        Opp_Install_Type)
            res=$(echo ${Opp_Install_Type})
            ;;
        Opp_Install_path_Param)
            res=$(echo ${Opp_Install_path_Param})
            ;;
        Opp_Version)
            res=$(echo ${Opp_Version})
            ;;
        Opp_Install_Path_Param)
            res=$(echo ${Opp_Install_Path_Param})
            ;;
        esac
    fi
    echo "${res}"
}

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"
function getRunpkgInfo() {
    local _key="$1"
    if [[ -f "${_VERSION_INFO_FILE}" ]]; then
        . "${_VERSION_INFO_FILE}"
        case "${_key}" in
        Version)
            echo ${Version}
            ;;
        esac
    fi
}

function updateInstallInfo() {
    local _key="$1"
    local _val="$2"
    local _is_new_gen="$3"
    local _old_val=$(getInstalledInfo "${_key}")
    local _target_install_info="${_TARGET_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
    if [[ -f "${_target_install_info}" ]]; then
        chmod 644 "${_target_install_info}"
        if [[ "${_old_val}"x == ""x ]] || [[ "${_is_new_gen}" == "true" ]]; then
            echo "${_key}=${_val}" >> "${_target_install_info}"
        else
            sed -i "/${_key}/c ${_key}=${_val}" "${_target_install_info}"
        fi
    else
        echo "${_key}=${_val}" > "${_target_install_info}"
    fi

    chmod 644 "${_target_install_info}" 2> /dev/null
    if [[ "$(id -u)" != "0" ]]; then
        chmod 600 "${_target_install_info}" 2> /dev/null
    fi
}

function updateInstallInfos() {
    local _uname="$1"
    local _ugroup="$2"
    local _type="$3"
    local _path="$4"
    local _version=$(getRunpkgInfo "${KEY_RUNPKG_VERSION}")
    local _target_install_info="${_TARGET_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
    local _is_new_gen="false"
    if [[ ! -f "${_target_install_info}" ]]; then
        _is_new_gen="true"
    fi
    updateInstallInfo "${KEY_INSTALLED_UNAME}" "${_uname}" "${_is_new_gen}"
    updateInstallInfo "${KEY_INSTALLED_UGROUP}" "${_ugroup}" "${_is_new_gen}"
    updateInstallInfo "${KEY_INSTALLED_TYPE}" "${_type}" "${_is_new_gen}"
    updateInstallInfo "${KEY_INSTALLED_PATH}" "${_path}" "${_is_new_gen}"
    updateInstallInfo "${KEY_INSTALLED_VERSION}" "${_version}" "${_is_new_gen}"
}

function checkFileExist() {
    local _path="$1"
    if [[ ! -f "${_path}" ]];then
        logAndPrint "ERR_NO:${FILE_READ_FAILED};ERR_DES:The file (${_path}) does not existed, install failed."
        return 1
    fi
    return 0
}

function createSoftLink() {
    local _src_path="$1"
    local _dst_path="$2"
    ln -s "${_src_path}" "${_dst_path}" 2> /dev/null
    if [[ "$?" != "0" ]]; then
        return 1
    else
        return 0
    fi
}

# init installation parameters
_TARGET_INSTALL_PATH="$1"
_TARGET_USERNAME="$2"
_TARGET_USERGROUP="$3"
install_type="$4"
is_quiet="$5"
_FIRST_NOT_EXIST_DIR="$6"
is_the_last_dir_opp="$7"
is_for_all="$8"
if [[ "${_TARGET_INSTALL_PATH}" == "" ]] || [[ "${_TARGET_USERNAME}" == "" ]] || 
[[ "${_TARGET_USERGROUP}" == "" ]] || [[ "${install_type}" == "" ]] || 
[[ "${is_quiet}" == "" ]]; then
    logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid for install."
    exit 1
fi

# init log file path
_INSTALL_INFO_FILE="${_TARGET_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
if [[ ! -f "${_INSTALL_INFO_FILE}" ]]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
fi

if [[ "$(id -u)" != "0" ]]; then
    _LOG_PATH_PERM="740"
    _LOG_FILE_PERM="640"
    _INSTALL_INFO_PERM="600"
    _LOG_PATH_AND_FILE_GROUP="root"
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH_PERM="750"
    _LOG_FILE_PERM="640"
    _INSTALL_INFO_PERM="644"
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

logAndPrint "[INFO]:Begin install opp module."
checkFileExist "${_FILELIST_FILE}"
if [[ "$?" != 0 ]]; then
    exit 1
fi

checkFileExist "${_COMMON_PARSER_FILE}"
if [[ "$?" != 0 ]]; then
    exit 1
fi

_BUILTIN_PERM="550"
_CUSTOM_PERM="750"
_CREATE_DIR_PERM="750"
_CREATE_FIRST_Ascend_DIR_PERM="755"
_ONLYREAD_PERM="440"
if [[ "${is_for_all}"=="y" ]]; then
    _BUILTIN_PERM="555"
    _CUSTOM_PERM="755"
    _CREATE_DIR_PERM="755"
    _CREATE_FIRST_Ascend_DIR_PERM="755"
    _ONLYREAD_PERM="444"
fi
if [[ "${_FIRST_NOT_EXIST_DIR}" != "" ]]; then
    mkdir -p "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
    chmod -R "${_CREATE_DIR_PERM}" "${_FIRST_NOT_EXIST_DIR}" 2> /dev/null
    chown -R "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${_FIRST_NOT_EXIST_DIR}" 2> /dev/null
    if [[ "$(id -u)" == "0" ]] && [[ "${is_the_last_dir_opp}" == "0" ]]; then
        chmod -R "${_CREATE_FIRST_Ascend_DIR_PERM}" "${_FIRST_NOT_EXIST_DIR}" 2> /dev/null
        chown -R "root":"root" "${_FIRST_NOT_EXIST_DIR}" 2> /dev/null
    fi
fi
# make the opp and the upper folder can write files
is_change_dir_mode="false"
if [[ "$(id -u)" != 0 ]] && [[ ! -w "${_TARGET_INSTALL_PATH}" ]]; then
    chmod u+w "${_TARGET_INSTALL_PATH}" 2> /dev/null
    is_change_dir_mode="true"
fi

# change installed folder's permission except aicpu
subdirs=$(ls "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null)
for dir in ${subdirs}; do
    if [[ ${dir} != "aicpu" ]] && [[ ${dir} != "script" ]]; then
        chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/opp/${dir}" 2> /dev/null
    fi
done
chmod "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null

logAndPrint "upgradePercentage:30%"
logAndPrint "[INFO]:Copy opp module source files to the install folder."

# copying opp source files
bash "${_COMMON_PARSER_FILE}" --copy --username="${_TARGET_USERNAME}" --usergroup="${_TARGET_USERGROUP}" "${install_type}" "${_TARGET_INSTALL_PATH}" "${_FILELIST_FILE}" 1> /dev/null
logWithErrorLevel "$?" "error" "ERR_NO:${INSTALL_FAILED};ERR_DES:Copy opp source files failed."

logAndPrint "[INFO]:Creating ("${_TARGET_INSTALL_PATH}""/ops") soft link from ("${_TARGET_INSTALL_PATH}""/opp")."
createSoftLink "${_TARGET_INSTALL_PATH}""/opp" "${_TARGET_INSTALL_PATH}""/ops"
logWithErrorLevel "$?" "warn" "[WARNING]:Create soft link for ops failed. That may \
cause some compatibility issues for old version envrionment."

if [[ "$(id -u)" == "0" ]]; then
    chown -h "root":"root" "${_TARGET_INSTALL_PATH}""/ops" 2> /dev/null
else
    chown -h "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${_TARGET_INSTALL_PATH}""/ops" 2> /dev/null
fi
logWithErrorLevel "$?" "warn" "[WARNING]:Change ops installed user or group failed. \
That may cause some compatibility issues for old version envrionment."
logAndPrint "upgradePercentage:50%"

logAndPrint "[INFO]:Copying version.info"
cp -f "${_VERSION_INFO_FILE}" "$_TARGET_INSTALL_PATH""/opp"
logWithErrorLevel "$?" "error" "ERR_NO:${INSTALL_FAILED};ERR_DES:Copy version.info file failed."

logAndPrint "[INFO]:Update the opp install info."
updateInstallInfos "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}" "${install_type}" "${_TARGET_INSTALL_PATH}"
logWithErrorLevel "$?" "error" "ERR_NO:${INSTALL_FAILED};ERR_DES:Update opp install info failed."

# change log dir and file owner and rights
chmod "${_LOG_PATH_PERM}" "${_LOG_PATH}" 2> /dev/null
chmod "${_LOG_FILE_PERM}" "${_INSTALL_LOG_FILE}" 2> /dev/null
chmod "${_LOG_FILE_PERM}" "${_OPERATE_LOG_FILE}" 2> /dev/null

# mkdir for custom ops
mkdir -p "${_TARGET_INSTALL_PATH}""/opp/framework/custom/" 2> /dev/null
mkdir -p "${_TARGET_INSTALL_PATH}""/opp/fusion_pass/custom/" 2> /dev/null
mkdir -p "${_TARGET_INSTALL_PATH}""/opp/fusion_rules/custom/" 2> /dev/null
mkdir -p "${_TARGET_INSTALL_PATH}""/opp/op_impl/custom/" 2> /dev/null
mkdir -p "${_TARGET_INSTALL_PATH}""/opp/op_proto/custom/" 2> /dev/null

# change installed folder's permission except aicpu
subdirs=$(ls "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null)
for dir in ${subdirs}; do
    if [[ ${dir} != "aicpu" ]] && [[ ${dir} != "script" ]]; then
        chmod -R "${_BUILTIN_PERM}" "${_TARGET_INSTALL_PATH}/opp/${dir}" 2> /dev/null
    fi
done

if [[ "$(id -u)" == "0" ]]; then
    chmod "755" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
else
    chmod "${_BUILTIN_PERM}" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
fi

chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}""/opp/framework/custom/" 2> /dev/null
chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}""/opp/fusion_pass/custom/" 2> /dev/null
chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}""/opp/fusion_rules/custom/" 2> /dev/null
chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}""/opp/op_impl/custom/" 2> /dev/null
chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}""/opp/op_proto/custom" 2> /dev/null

chmod "${_ONLYREAD_PERM}" "${_TARGET_INSTALL_PATH}""/opp/scene.info" 2> /dev/null
chmod "${_ONLYREAD_PERM}" "${_TARGET_INSTALL_PATH}""/opp/version.info" 2> /dev/null
chmod 600 "${_TARGET_INSTALL_PATH}""/opp/ascend_install.info" 2> /dev/null

if [[ "${is_change_dir_mode}" == "true" ]]; then
    chmod u-w "${_TARGET_INSTALL_PATH}" 2> /dev/null
fi

# change installed folder's owner and group except aicpu
subdirs=$(ls "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null)
for dir in ${subdirs}; do
    if [[ ${dir} != "aicpu" ]] && [[ ${dir} != "script" ]]; then
        chown -R "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${_TARGET_INSTALL_PATH}/opp/${dir}" 2> /dev/null
    fi
done
chown "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
logWithErrorLevel "$?" "error" "ERR_NO:${INSTALL_FAILED};ERR_DES:Change opp onwership failed.."

logAndPrint "upgradePercentage:100%"
logAndPrint "[INFO]:Opp package install success! The new version takes effect immediately."

logAndPrint "[INFO]:Installation information listed below:"
logAndPrint "[INFO]:Install path: (${_TARGET_INSTALL_PATH}/opp)"
logAndPrint "[INFO]:Install log file path: (${_INSTALL_LOG_FILE})"
logAndPrint "[INFO]:Operation log file path: (${_OPERATE_LOG_FILE})"
logAndPrint "[INFO]:Using requirements: when opp module install finished or \
before you run the opp module, execute the command
[ export ASCEND_OPP_PATH=${_TARGET_INSTALL_PATH}/opp ] to set the environment path."
exit 0
