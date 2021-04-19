#!/bin/bash

# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_FILELIST_FILE="${_CURR_PATH}""/../../filelist.csv"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"
_VERSION_INFO_FILE="${_CURR_PATH}""/../../version.info"
_INSTALL_SHELL_FILE="${_CURR_PATH}""/opp_install.sh"

_INSTALL_LOG_DIR="/opp/install_log"
_INSTALL_INFO_SUFFIX="/opp/ascend_install.info"

FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
UPGRADE_FAILED="0x0000"
UPGRADE_FAILED_DES="Update successed."

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

function checkFolderExist() {
    local _path="${1}"
    if [[ ! -d "${_path}" ]]; then
        logAndPrint "ERR_NO:${FILE_READ_FAILED};ERR_DES:Installation directroy \
[${_path}] does not exist, upgrade failed."
        exit 1
    fi
}

function checkFileExist() {
    local _path="${1}"
    if [[ ! -f "${_path}" ]];then
        logAndPrint "ERR_NO:${FILE_READ_FAILED};ERR_DES:The file (${_path}) \
does not existed, upgrade failed."
        exit 1
    fi
}

# user check functions
check_user(){
    local _uname="$1"
    ret=`cat /etc/passwd | cut -f1 -d':' | grep -w "${_uname}" -c`
    if [[ $ret -le 0 ]]; then
        return 1
    else
        return 0
    fi
}

check_group(){
    local _ugroup="$1"
    local _uname="$2"
    group_user_related=`groups "${_uname}"|awk -F":" '{print $2}'|grep -w "${_ugroup}"`
    if [[ "${group_user_related}x" != "x" ]];then
        return 0
    else
        return 1
    fi
}

# check user name and user group is valid or not
checkInstallUserGroupConditon() {
    local _uname="$1"
    local _ugroup="$2"
    check_user "${_uname}"
    if [[ $? -ne 0 ]];then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Username ${_uname} not \
exists! Please add ${_uname} user."
        exit 1
    fi
    check_group "${_ugroup}" "${_uname}"
    if [[ $? -ne 0 ]];then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Usergroup ${_ugroup} \
not right! Please check the relatianship of user ${_uname} and the group ${_ugroup}."
        exit 1
    fi
}

checkInstalledType() {
    local _type="$1"
    if [[ "${_type}" != "run" ]] && 
    [[ "${_type}" != "full" ]] && 
    [[ "${_type}" != "devel" ]]; then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Install type \
[${_ugroup}] of opp module is not right!"
        exit 1
    fi
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

# init input paremeters
_TARGET_INSTALL_PATH="$1"
_TARGET_USERNAME="$2"
_TARGET_USERGROUP="$3"
is_quiet="$4"

# check input parameters is valid
if [[ "${_TARGET_INSTALL_PATH}" == "" ]] || [[ "${_TARGET_USERNAME}" == "" ]] ||
[[ "${_TARGET_USERGROUP}" == "" ]] || [[ "${is_quiet}" == "" ]]; then
    logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid for upgrade."
    exit 1
fi

# init log file path
_UNINSTALL_SHELL_FILE="${_TARGET_INSTALL_PATH}""/opp/script/opp_uninstall.sh"
# adpter for old version's path
if [[ ! -f "${_UNINSTALL_SHELL_FILE}" ]]; then
    _UNINSTALL_SHELL_FILE="${_TARGET_INSTALL_PATH}""/opp/scripts/opp_uninstall.sh"
fi
_INSTALL_INFO_FILE="${_TARGET_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
_IS_ADAPTER_MODE="false"
if [[ ! -f "${_INSTALL_INFO_FILE}" ]]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
    _IS_ADAPTER_MODE="true"
fi

if [[ "$(id -u)" != "0" ]]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

_TARGET_USERNAME=$(getInstalledInfo "${KEY_INSTALLED_UNAME}")
_TARGET_USERGROUP=$(getInstalledInfo "${KEY_INSTALLED_UGROUP}")

# check install conditons by specific install path
install_type=$(getInstalledInfo "${KEY_INSTALLED_TYPE}")
if [[ "${install_type}" == "" ]]; then
    logWithErrorLevel "1" "error" "ERR_NO:${UPGRADE_FAILED};ERR_DES:Opp module\
 is not installed or directory is wrong."
fi
checkInstallUserGroupConditon "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
checkInstalledType "${install_type}"
checkFileExist "${_FILELIST_FILE}"
checkFileExist "${_COMMON_PARSER_FILE}"
# check the opp module sub directory exist or not
opp_sub_dir="${_TARGET_INSTALL_PATH}""/opp/"
checkFolderExist "${opp_sub_dir}"
logAndPrint "upgradePercentage:10%"

logAndPrint "[INFO]:Begin upgrade opp module."
logAndPrint "[INFO]:Uninstall the Existed opp module before upgrade."
# call uninstall functions
if [[ "${_IS_ADAPTER_MODE}" == "true" ]]; then
    if [[ ! -f "${_UNINSTALL_SHELL_FILE}" ]]; then
        logAndPrint "ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_UNINSTALL_SHELL_FILE}) not exists. Please make sure that the opp module\
 installed in (${_TARGET_INSTALL_PATH}) and then set the correct install path."
        exit 1
    fi
    bash "${_UNINSTALL_SHELL_FILE}" "${install_path}" "uninstall"
    _IS_FRESH_ISNTALL_DIR="1"
    bash "${_INSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}" "${install_type}" "${is_quiet}" "${_IS_FRESH_ISNTALL_DIR}"
    if [[ "$?" == 0 ]]; then
        exit 0
    else
        exit 1
    fi
else
    bash "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "upgrade" "${is_quiet}"
fi

_BUILTIN_PERM="550"
_CUSTOM_PERM="750"
_ONLYREAD_PERM="440"
if [[ "$(id -u)" != "0" ]]; then
    _INSTALL_INFO_PERM="600"
else
    _INSTALL_INFO_PERM="644"
fi
# change permission for install folders
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

logWithErrorLevel "$?" "error" "ERR_NO:${UPGRADE_FAILED};ERR_DES:Uninstall the \
installed directory (${_TARGET_INSTALL_PATH}) failed."
logAndPrint "upgradePercentage:30%"
# copy module source files
bash ${_COMMON_PARSER_FILE} --copy --username="${_TARGET_USERNAME}" --usergroup="${_TARGET_USERGROUP}" "${install_type}" "${_TARGET_INSTALL_PATH}" "${_FILELIST_FILE}" 1> /dev/null
logWithErrorLevel "$?" "error" "ERR_NO:${UPGRADE_FAILED};ERR_DES:Copy opp source files failed."
logAndPrint "upgradePercentage:50%"

logAndPrint "[INFO]:Copying version.info"
cp -f "${_VERSION_INFO_FILE}" "$_TARGET_INSTALL_PATH""/opp"
logWithErrorLevel "$?" "error" "ERR_NO:${INSTALL_FAILED};ERR_DES:Copy version.info file failed."

# create ops soft link and change ownership
logAndPrint "[INFO]:Creating ("${_TARGET_INSTALL_PATH}""/ops") soft link from ("${_TARGET_INSTALL_PATH}""/opp")"
createSoftLink "${_TARGET_INSTALL_PATH}/opp" "${_TARGET_INSTALL_PATH}/ops"
logWithErrorLevel "$?" "warn" "[WARNING]:Create soft link for ops failed. That may \
cause some compatibility issues for old version envrionment."
chown -h "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${_TARGET_INSTALL_PATH}""/ops" 2> /dev/null
logWithErrorLevel "$?" "warn" "[WARNING]:Change ops installed user or group failed. \
That may cause some compatibility issues for old version envrionment."

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
logAndPrint "[INFO]:Upgrade opp module success."

logAndPrint "[INFO]:Installation information listed below:"
logAndPrint "[INFO]:Install path: (${_TARGET_INSTALL_PATH}/opp)"
logAndPrint "[INFO]:Install log file path: (${_INSTALL_LOG_FILE})"
logAndPrint "[INFO]:Operation log file path: (${_OPERATE_LOG_FILE})"
logAndPrint "[INFO]:Using requirements: when opp module install finished or \
before you run the opp module, execute the command \
[ export ASCEND_OPP_PATH=${_TARGET_INSTALL_PATH}/opp ] to set the environment path."
exit 0
