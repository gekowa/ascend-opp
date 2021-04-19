#!/bin/bash
PARAM_INVALID="0x0002"
PARAM_INVALID_DES="Invalid input parameter."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
OPERATE_FAILED="0x0001"

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

function checkDirectoryExist() {
    local _path="${1}"
    if [[ ! -d "${_path}" ]]; then
        logAndPrint "ERR_NO:${FILE_READ_FAILED};ERR_DES:Installation directroy [${_path}] does not exist, uninstall failed."
        return 1
    else
        return 0
    fi
}

function checkFileExist() {
    local _path="${1}"
    if [[ ! -f "${_path}" ]];then
        logAndPrint "ERR_NO:${FILE_READ_FAILED};ERR_DES:The file (${_path}) does not existed."
        return 1
    else
        return 0
    fi
}

# DFS sub-folders cleaner
function deleteEmptyFolders() {
    local _init_dir="$1"
    local _aicpu_filter="$2"
    find "${_init_dir}" -mindepth 1 -maxdepth 1 -type d ! \
        -path "${_aicpu_filter}" 2> /dev/null | while read -r dir
    do
        if [[ "$(echo "${dir}" | grep "custom")" == "" ]]; then
            deleteEmptyFolders "${dir}"

            if [[ "$(find "${dir}" -mindepth 1 -type d)" == "" ]] && \
                [[ "$(ls -A "${dir}")" == "" ]] >/dev/null; then
                rm -rf "${dir}"
            fi
        else
            # remove custom folders which not contains sub-folder or any files
            if [[ "$(ls -A "${dir}")" == "" ]]; then
                rm -rf "${dir}"
            fi
        fi
    done
}

function checkInstalledType() {
    local _type="$1"
    if [[ "${_type}" != "run" ]] &&
    [[ "${_type}" != "full" ]] &&
    [[ "${_type}" != "devel" ]]; then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Install type \
[${_ugroup}] of opp module is not right!"
        return 1
    else
        return 0
    fi
}

installed_path="$1"
uninstall_mode="$2"
is_quiet="$3"
paramter_num="$#"

if [[ "${paramter_num}" != 0 ]]; then
    if [[ "${installed_path}" == "" ]] ||
    [[ "${uninstall_mode}" == "" ]] ||
    [[ "${is_quiet}" == "" ]] ; then
        logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid\
for call uninstall functions."
        exit 1
    fi
fi

_CURR_PATH=$(dirname $(readlink -f $0))
_FILELIST_FILE="${_CURR_PATH}""/filelist.csv"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"
_TARGET_INSTALL_PATH="${_CURR_PATH}""/../.."
_INSTALL_LOG_DIR="/opp/install_log"
_INSTALL_INFO_SUFFIX="/opp/ascend_install.info"
_VERSION_INFO_SUFFIX="/opp/version.info"
# avoid relative path casued errors by delete floders
_ABS_INSTALL_PATH=$(cd ${_TARGET_INSTALL_PATH}; pwd)
# init log file path
_INSTALL_INFO_FILE="${_ABS_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
if [[ ! -f "${_INSTALL_INFO_FILE}" ]]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
fi

_VERSION_INFO_FILE="${_ABS_INSTALL_PATH}${_VERSION_INFO_SUFFIX}"

if [[ "$(id -u)" != "0" ]]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

logAndPrint "[INFO]:Begin uninstall the opp module."

# check install folder existed
checkFileExist "${_INSTALL_INFO_FILE}"
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."
checkFileExist "${_FILELIST_FILE}"
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."
checkFileExist "${_COMMON_PARSER_FILE}"
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."
opp_sub_dir="${_ABS_INSTALL_PATH}""/opp"
checkDirectoryExist "${opp_sub_dir}"
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."

installed_type=$(getInstalledInfo "${KEY_INSTALLED_TYPE}")
checkInstalledType "${installed_type}"
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."

_CUSTOM_PERM="755"
_BUILTIN_PERM="555"
# make the opp and the upper folder can write files
is_change_dir_mode="false"
if [[ "$(id -u)" != 0 ]] && [[ ! -w "${_TARGET_INSTALL_PATH}" ]]; then
    chmod u+w "${_TARGET_INSTALL_PATH}" 2> /dev/null
    is_change_dir_mode="true"
fi

# change installed folder's permission except aicpu
subdirs=$(ls "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null)
for dir in ${subdirs}; do
    if [[ ${dir} != "aicpu" ]]; then
        chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/opp/${dir}" 2> /dev/null
    fi
done
chmod "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null

# delete soft link of ops
ops_soft_link="${_ABS_INSTALL_PATH}""/ops"
logAndPrint "[INFO]:Delete the ops soft link (${ops_soft_link})."
rm -rf "${ops_soft_link}"
logWithErrorLevel "$?" "warn" "[WARNING]Delete ops soft link failed, that may cause \
some error to old version opp module."

# delete opp source files
logAndPrint "[INFO]:Delete the installed opp source files in (${_ABS_INSTALL_PATH})."
bash "${_COMMON_PARSER_FILE}" --remove "${installed_type}" "${_ABS_INSTALL_PATH}" "${_FILELIST_FILE}" 1> /dev/null
logWithErrorLevel "$?" "error" "ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall opp module failed."

# delete version.info file
logAndPrint "[INFO]:Delete the version info file (${_VERSION_INFO_FILE})."
rm -f "${_VERSION_INFO_FILE}"
logWithErrorLevel "$?" "warn" "[WARNING]Delete opp version info file failed, \
please delete it by yourself."

# delete install.info file
if [[ "${uninstall_mode}" != "upgrade" ]]; then
    logAndPrint "[INFO]:Delete the install info file (${_INSTALL_INFO_FILE})."
    rm -f "${_INSTALL_INFO_FILE}"
    logWithErrorLevel "$?" "warn" "[WARNING]Delete ops install info file failed, \
please delete it by yourself."
fi

# delete *.pyc files
pyc_path=$(find "${opp_sub_dir}/op_impl/built-in/ai_core/tbe/impl" -name "__pycache__" 2> /dev/null)
for var in ${pyc_path[@]}
    do
    rm -rf "${var}" 2> /dev/null
    done

# delete the emtpy folders
aicpu_filter="${opp_sub_dir}/aicpu"
# deleteEmptyFolders "${opp_sub_dir}" "${aicpu_filter}"

# delete the empty opp folder it'self
res=$(ls "${opp_sub_dir}")
if [[ "${res}" == "" ]]; then
    rm -rf "${opp_sub_dir}" >> /dev/null 2>&1
fi

# delete the upper folder when it is empty
dir_existed=$(ls "${_ABS_INSTALL_PATH}" 2> /dev/null)
if [[ "${dir_existed}" == "" ]] && [[ "${uninstall_mode}" != "upgrade" ]]; then
    rm -rf "${_ABS_INSTALL_PATH}" >> /dev/null 2>&1
fi

# change installed folder's permission except aicpu
subdirs=$(ls "${_ABS_INSTALL_PATH}/opp" 2> /dev/null)
for dir in ${subdirs}; do
    if [[ ${dir} != "aicpu" ]]; then
        chmod "${_BUILTIN_PERM}" "${_ABS_INSTALL_PATH}/opp/${dir}" 2> /dev/null
    fi
done
chmod "${_BUILTIN_PERM}" "${_ABS_INSTALL_PATH}/opp" 2> /dev/null

if [[ "${is_change_dir_mode}" == "true" ]]; then
    chmod u-w "${_ABS_INSTALL_PATH}" 2> /dev/null
fi

if [[ -d "${_ABS_INSTALL_PATH}/opp/" ]]; then
# find custom file in path and print log
    for file in ${_ABS_INSTALL_PATH}/opp/*
    do
        logAndPrint "[WARN]: ${file}, Has files changed by users, cannot be delete."
    done
fi

logAndPrint "[INFO]:Opp package uninstall success! Uninstallation takes effect immediately."
exit 0