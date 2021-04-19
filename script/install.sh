#!/bin/bash

_CURR_OPERATE_USER="$(id -nu 2> /dev/null)"
_CURR_OPERATE_GROUP="$(id -ng 2> /dev/null)"
_DEFAULT_USERNAME="HwHiAiUser"
_DEFAULT_USERGROUP="HwHiAiUser"
_DEFAULT_INSTALL_PATH="/usr/local/Ascend"
# defaults for general user
if [[ "$(id -u)" != "0" ]]; then
    _DEFAULT_USERNAME="${_CURR_OPERATE_USER}"
    _DEFAULT_USERGROUP="${_CURR_OPERATE_GROUP}"
    _DEFAULT_INSTALL_PATH="/home/${_CURR_OPERATE_USER}/Ascend"
fi

# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_INSTALL_SHELL_FILE="${_CURR_PATH}""/opp_install.sh"
_UPGRADE_SHELL_FILE="${_CURR_PATH}""/opp_upgrade.sh"
_RUN_PKG_INFO_FILE="${_CURR_PATH}""/../scene.info"
_VERSION_INFO_FILE="${_CURR_PATH}""/../../version.info"
_COMMON_INC_FILE="${_CURR_PATH}""/common_func.inc"
_VERCHECK_FILE="${_CURR_PATH}""/ver_check.sh"

# defaluts info determinated by user's inputs
_INSTALL_LOG_DIR="/opp/install_log"
_INSTALL_INFO_SUFFIX="/opp/ascend_install.info"
_VERSION_INFO_SUFFIX="/opp/version.info"
_TARGET_INSTALL_PATH=""
_TARGET_USERNAME=""
_TARGET_USERGROUP=""

# error number and description
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
FILE_WRITE_FAILED="0x0081"
FILE_WRITE_FAILED_DES="File write failed."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
FILE_REMOVE_FAILED="0x0090"
FILE_REMOVE_FAILED_DES="Failed to remove file."
UNAME_NOT_EXIST="0x0091"
UNAME_NOT_EXIST_DES="Username not exists."
OS_CEHCK_ERR="0x0092"
OS_CEHCK_ERR_DES="OS check error."
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

# user check functions
function checkUserExist(){
    local _uname="$1"
    ret=`cat /etc/passwd | cut -f1 -d':' | grep -w "${_uname}" -c`
    if [[ "${ret}" -le 0 ]]; then
        return 1
    else
        return 0
    fi
}

function checkGroupValidWithUser(){
    local _ugroup="$1"
    local _uname="$2"
    local _related=`groups "${_uname}" 2> /dev/null |awk -F":" '{print $2}'|grep -w "${_ugroup}"`
    if [[ "${_related}" != "" ]];then
        return 0
    else
        return 1
    fi
}

# check user name and user group is valid or not
function checkUserGroupRelationValid() {
    local _uname="$1"
    local _ugroup="$2"
    if [[ $_uname == "" ]] || [[ $_ugroup == "" ]]; then
        logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Input empty username or usergroup is invalid."
        return 1
    fi
    checkUserExist "${_uname}"
    if [[ $? -ne 0 ]];then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Username ${_uname} not exists!"
        return 1
    fi
    checkGroupValidWithUser "${_ugroup}" "${_uname}"
    if [[ $? -ne 0 ]];then
        logAndPrint "ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Usergroup ${_ugroup} \
not right! Please check the relatianship of user ${_uname} and the group ${_ugroup}."
        return 1
    fi
    return 0
}

function checkOSAndArch() {
    local _pkg_os="unknow"
    local _pkg_arch="unknow"
    if [[ -f "${_RUN_PKG_INFO_FILE}" ]]; then
        . "${_RUN_PKG_INFO_FILE}"
        _pkg_os="${os}"
        _pkg_arch="${arch}"
    else
        logAndPrint "[WARNING]:Version info file of this run package is not existed."
    fi

    . "${_COMMON_INC_FILE}"
    get_system_info
    local _env_arch="unknow"
    local _env_os="unknow"
    local _host_arch="${HostArch}"
    if [[ "$(echo ${_host_arch} | grep -i "x86")" != "" ]]; then
        _env_arch="x86_64"
    elif [[ "$(echo ${_host_arch} | grep -i "aarch64")" != "" ]]; then
        _env_arch="aarch64"
    fi

    local _host_os_name="${HostOsName}"
    if [[ "$(echo ${_host_os_name} | grep -i "euler")" != "" ]]; then
        _env_os="euleros"
    elif [[ "$(echo ${_host_os_name} | grep -i "ubuntu")" != "" ]]; then
        _env_os="ubuntu"
    elif [[ "$(echo ${_host_os_name} | grep -i "debian")" != "" ]]; then
        _env_os="debian"
    else
        if [[ "$(echo ${_host_os_name} | grep -i "cent")" != "" ]] ||
        [[ "$(echo ${_host_os_name} | grep -i "red hat")" != "" ]]; then
            _env_os="centos"
        fi
    fi

    if [[ "${_env_arch}" == "${_pkg_arch}" ]] && [[ "${_env_os}" == "${_pkg_os}" ]]; then
        logAndPrint "[INFO]:OS check passed. Install for ${_env_arch} ${_env_os} system."
        return 0
    else
        if [[ "${is_quiet}" == y ]]; then
            logAndPrint "[WARNING]:The operating system is ${_env_arch} ${_env_os},\
 but the run package is for ${_pkg_arch} ${_pkg_os}. Are you sure to keep operating? y"
        else
            logAndPrint "[WARNING]:The operating system is ${_env_arch} ${_env_os},\
 but the run package is for ${_pkg_arch} ${_pkg_os}. Are you sure to keep operating? [y/n]"
            while true
            do
                read yn
                if [[ "$yn" == n ]]; then
                    logAndPrint "[INFO]:Exit to install opp module."
                    return 1
                elif [ "$yn" = y ]; then
                    break;
                else
                    echo "[WARNING]:Input error, please input y or n to choose!"
                fi
            done
        fi
    fi
    return 0
}

function checkEmtpyUser() {
    local _cmd_type="$1"
    local _uname="$2"
    local _ugroup="$3"
    if [[ "${_uname}" != "" ]] || [[ "${_ugroup}" != "" ]]; then
        logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Operation of \
${_cmd_type} opp not support specific user name or user group. Please \
use [--help] to see the useage."
        return 1
    fi
}

function logOperationRetStatus() {
    local _cmd_type="$1"
    local _install_type="$2"
    local _ret_status="$3"
    local _cmd_list="$*"
    _cmd_list=$(echo $_cmd_list | awk -F" " '{$1="";$2="";$3=""; print $0}' | awk -F"   " '{print $2}')

    local _event_level="SUGGESTION"
    if [[ "${_cmd_type}" == "upgrade" ]]; then
        _event_level="MINOR"
    elif [[ "${_cmd_type}" == "uninstall" ]]; then
        _event_level="MAJOR"
    fi
    local _ret_status_des="success"
    if [[ "${_ret_status}" != 0 ]]; then
        _ret_status_des="failed"
    fi

    local _curr_user="${_CURR_OPERATE_USER}"
    local _curr_ip="127.0.0.1"
    local _pkg_name="OPP"
    local _cur_date=$(getDate)
    echo "Install ${_event_level} ${_curr_user} ${_cur_date} ${_curr_ip} \
${_pkg_name} ${_ret_status_des} install_type=${_install_type}; \
cmdlist=${_cmd_list}."
    if [[ -f "${_OPERATE_LOG_FILE}" ]]; then
        echo "Install ${_event_level} ${_curr_user} ${_cur_date} ${_curr_ip} \
${_pkg_name} ${_ret_status_des} install_type=${_install_type}; \
cmdlist=${_cmd_list}." >> "${_OPERATE_LOG_FILE}"
    else
        echo "[WARNING]: Operation log file not exist."
    fi

    if [[ "${_ret_status}" != 0 ]]; then
        exit 1
    else
        exit 0
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

function precleanBeforeInstall() {
    local _path="$1"
    if [[ "${_path}" == "" ]]; then
        logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:Input empty path is invalid."
        return 1
    fi
    local _opp_sub_dir="${_path}""/opp"
    local _installed_path=$(getInstalledInfo "${KEY_INSTALLED_PATH}")
    local _files_existed=1
    # check the installation folder has files or opp module existed or not
    local _existed_files=$(find ${_opp_sub_dir} -path ${_opp_sub_dir}/aicpu -prune -o -type f -print 2> /dev/null)
    if [[ "${_existed_files}" == "" ]]; then
        _files_existed=1
    else
        _files_existed=0
        logAndPrint "[WARNING]:Install folder has files existed. Some files are listed below:"
        _ret_array=(`echo ${_existed_files}|awk '{len=split($0,arr," ");for(i=1;i<=len;i++) print arr[i]}'`)
        for idx in {0..5}; do
            if [[ "${_ret_array[$idx]}" != "" ]]; then
                logAndPrint "file: "${_ret_array[$idx]}
            fi
        done
    fi

    if [[ "${_files_existed}" == "0" ]]; then
        if [[ "${is_quiet}" == y ]]; then
            logAndPrint "[WARNING]:Directory has file existed or installed opp \
module, are you sure to keep installing opp module in it? y"
        else
            if [[ ! -f  ${_opp_sub_dir}"/ascend_install.info" ]]; then
                logAndPrint "Directory has file existed, do you want to continue? [y/n]"
            else
                logAndPrint "Opp package has been installed on the path $(getInstalledInfo "${KEY_INSTALLED_PATH}"), \
the version is $(getInstalledInfo "${KEY_INSTALLED_VERSION}"), \
and the version of this package is $(getRunpkgInfo "${KEY_RUNPKG_VERSION}"), do you want to continue? [y/n]"
            fi
            while true
            do
                read yn
                if [[ "$yn" == n ]]; then
                    logAndPrint "[INFO]:Exit to install opp module."
                    exit 0
                elif [ "$yn" = y ]; then
                    break;
                else
                    echo "[WARNING]:Input error, please input y or n to choose!"
                fi
            done
        fi
    else
        logAndPrint "[INFO]:Directory is empty, directly install opp module."
    fi

    logAndPrint "[INFO]:Directory is empty, directly install opp module."
    local ret=0
    if [ $(id -u) -eq 0 ]; then
        parent_dirs_permission_check "${_path}" && ret=$? || ret=$?
        if [ ${is_quiet} = "y" ]; then
            if [ ${ret} -eq 3 ]; then
                # permission > 755, pass, but pring warn
                logAndPrint "[WARNING]: You are going to put run-files on a unsecure install-path! \
Please change permission after install."
            elif [ ${ret} -ne 0 ]; then
                logAndPrint "ERR_NO:0x0095;ERR_DES:the given dir, or its parents, permission is invalid. \
Please install without quiet mode and check permission."
                exit 1
            fi
        else
            if [ ${ret} -ne 0 ]; then
                logAndPrint "[WARN]:You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        exit 1
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        echo "ERR_NO:0x0002;ERR_DES:input error, please input again!"
                    fi
                done
            fi
        fi
    fi

    if [[ "${_files_existed}" == "0" ]] && [[ "${_installed_path}" == "${_path}" ]]; then
        logAndPrint "[INFO]:Clean the installed opp module before install."
        if [[ ! -f "${_UNINSTALL_SHELL_FILE}" ]]; then
            logAndPrint "ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_UNINSTALL_SHELL_FILE}) not exists. Please set the correct install \
path or clean the previous version opp install info (/etc/ascend_install.info) and then reinstall it."
            return 1
        fi
        bash "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "uninstall" "${is_quiet}"
        if [[ "$?" != 0 ]]; then
            logAndPrint "ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
            return 1
        fi
    fi
}

function checkEmptyUserAndGroup() {
    local _username="$1"
    local _usergroup="$2"
    if [[ $(id -u) -ne 0 ]]; then
        if [[ "${_username}" != "" ]] || [[ "${_usergroup}" != "" ]]; then
            logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:\
Only root user can specific user name or user group."
            return 1
        else
            return 0
        fi
    else
        return 0
    fi
}


parent_dirs_permission_check() {
    current_dir="$1" parent_dir="" short_install_dir=""
    local owner="" mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")
    logAndPrint "[INFO]:parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

    if [ "${current_dir}"x = "/"x ]; then
        logAndPrint "[INFO]:parent_dirs_permission_check success"
        return 0
    else
        owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
        if [ "${owner}" != "root" ]; then
            logAndPrint "[WARNING]:[${short_install_dir}] permission isn't right, it should belong to root."
            return 1
        fi
        logAndPrint "[INFO]:[${short_install_dir}] belongs to root."

        mod_num=$(stat -c %a "${parent_dir}"/"${short_install_dir}")
        if [ ${mod_num} -lt 755 ]; then
            logAndPrint "[WARNING]:[${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
            return 2
        elif [ ${mod_num} -eq 755 ]; then
            logAndPrint "[INFO]:[${short_install_dir}] permission is ok."
        else
            logAndPrint "[WARNING]:[${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
            [ ${is_quiet} = n ] && return 3
        fi

        parent_dirs_permission_check "${parent_dir}"
    fi
}

function getOperationName() {
    local _cmd_name="install"
    if [[ "${is_upgrade}" == "y" ]]; then
        _cmd_name="upgrade"
    elif [[ "${is_uninstall}" == "y" ]]; then
        _cmd_name="uninstall"
    fi
    echo "${_cmd_name}"
}

function getOperationInstallType() {
    local _cmd_name=$(getOperationName)
    local _cmd_install_type="${in_install_type}"
    if [[ "${_cmd_name}" != "install" ]]; then
        _cmd_install_type=$(getInstalledInfo "${KEY_INSTALLED_TYPE}")
    fi
    echo "${_cmd_install_type}"
}

function getFirstNotExistDir() {
    local in_tmp_dir="$1"
    local arr=(`echo ${in_tmp_dir} | awk '{len=split($0,arr,"/");for(i=1;i<=len;i++) print arr[i]}'`)
    local len="${#arr[*]}"
    local tmp_dir[0]=""
    local tmp_str=""
    for((i=0;i<"${len}";i++)); do
        tmp_str="${tmp_str}""/""${arr["$i"]}"
        tmp_dir["$i"]="${tmp_str}"
    done
    for((i=0;i<"${len}";i++)); do
        if [[ ! -d "${tmp_dir["$i"]}" ]]; then
            echo "${tmp_dir["$i"]}"
            break
        fi
    done
}

function isOnlyLastNotExistDir() {
    local in_tmp_dir="$1"
    local arr=(`echo ${in_tmp_dir} | awk '{len=split($0,arr,"/");for(i=1;i<=len;i++) print arr[i]}'`)
    local len="${#arr[*]}"
    local tmp_dir[0]=""
    local tmp_str=""
    tmp_dir["0"]="/"
    for((i=1;i<="${len}";i++)); do
        tmp_str="${tmp_str}""/""${arr["$i-1"]}"
        tmp_dir["$i"]="${tmp_str}"
    done
    less_len=$((len))
    for((i=0;i<less_len;i++)); do
        if [[ ! -d "${tmp_dir["$i"]}" ]]; then
            isOnlyLastNotExistDir_path=""
            The_Num_Last_Not_ExistDir="${tmp_dir["$i"]}"
            break
        else
            isOnlyLastNotExistDir_path="${tmp_dir["$i"]}"
        fi
    done
    return
}

function matchFullpath() {
    echo "$(cd ${1%/*}; pwd)/${1##*/}"
    return
}

function checkPreFoldersPermission() {
    local in_tmp_dir="$1"
    local _uname="$2"
    local arr=(`echo ${in_tmp_dir} | awk '{len=split($0,arr,"/");for(i=1;i<=len;i++) print arr[i]}'`)
    local len="${#arr[*]}"
    local tmp_dir[0]=""
    local tmp_str=""
    for((i=0;i<"${len}";i++)); do
        tmp_str="${tmp_str}""/""${arr["$i"]}"
        tmp_dir["$i"]="${tmp_str}"
    done
    for((i=0;i<"${len}";i++)); do
        if [[ -d "${tmp_dir["$i"]}" ]]; then
            if [[ "$(id -u)" == "0" ]]; then
                su - "${_uname}" -c "cd ${tmp_dir["$i"]} >> /dev/null 2>&1"
                if [[ "$?" != "0" ]]; then
                    logAndPrint "ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname} do \
not have the permission to access ${tmp_dir["$i"]}, please reset the directory \
to a right permission."
                    return 1
                fi
            else
                # general user only can install for himself
                cd ${tmp_dir["$i"]} >> /dev/null 2>&1
                if [[ "$?" != "0" ]]; then
                    logAndPrint "ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname} do \
not have the permission to access ${tmp_dir["$i"]}, please reset the directory \
to a right permission."
                    return 1
                fi
                cd - >> /dev/null 2>&1
            fi
        fi
    done
    return 0
}

function checkLastFoldersPermissionforcheckpath() {
    local in_tmp_dir="$1"
    local _uname="$2"

    if [[ -d "${in_tmp_dir}" ]]; then
        if [[ "$(id -u)" == "0" ]]; then
            su - "${_uname}" -c "test -w ${tmp_dir["$i"]} >> /dev/null 2>&1"
            if [[ "$?" != "0" ]]; then
                logAndPrint "ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname} do \
access ${in_tmp_dir} failed, please reset the directory \
to a right permission."
                return 1
            fi
        else
                # general user only can install for himself
            test -w ${in_tmp_dir} >> /dev/null 2>&1
            if [[ "$?" != "0" ]]; then
                logAndPrint "ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname} \
access ${in_tmp_dir} failed, please reset the directory \
to a right permission."
                return 1
            fi
        fi
    fi
    return 0
}

function creat_checkpath() {
    created_path=$1
    _target_username=$2
    _target_usergroup=$3
    mkdir "${created_path}"
    if [[ "$(id -u)" != "0" ]]; then
        chmod 750 "${created_path}" 2> /dev/null
        chown "${_target_username}":"${_target_usergroup}" "${created_path}" 2> /dev/null
    else
        chmod 755 "${created_path}" 2> /dev/null
        chown root:root "${created_path}" 2> /dev/nul
    fi

    ret_created_path=${created_path}
    return
}


function select_last_dir_component (){
    path="$1"
    last_component=$(basename ${path})
    if [[ "${last_component}" == "atc" ]] ;then
        last_component="atc"
        return
    elif [[ "${last_component}" == "fwkacllib" ]]; then
        last_component="fwkacllib"
        return
    else
        last_component="atc or fwkacllib"
        return
    fi
}

function check_version_file () {
    pkg_path="$1"
    component_ret="$2"
    run_pkg_path_temp=$(dirname "${pkg_path}")
    run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
    version_file="${run_pkg_path}""/version.info"
    if [ -f "${version_file}" ];then
        echo "${version_file}" >> /dev/null 2
    else
        logAndPrint "ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
        exit 1
    fi
    return
}

function check_opp_version_file () {
    if [ -f "${_CURR_PATH}/../../version.info" ];then
        opp_ver_info="${_CURR_PATH}/../../version.info"
    elif [ -f "${_DEFAULT_INSTALL_PATH}/opp/version.info" ];then
        opp_ver_info="${_DEFAULT_INSTALL_PATH}/opp/version.info"
    else
        logAndPrint "ERR_NO:${FILE_NOT_EXIST}; The [opp] version.info not exists."
        exit 1
    fi
    return
}

function check_relation () {
    opp_ver_info="$1"
    req_pkg_name="$2"
    req_pkg_version="$3"
    _COMMON_INC_FILE="${_CURR_PATH}/common_func.inc"
    if [ -f "${_COMMON_INC_FILE}" ];then
    . "${_COMMON_INC_FILE}"
    check_pkg_ver_deps "${opp_ver_info}" "${req_pkg_name}" "${req_pkg_version}"
    ret_situation=$VerCheckStatus
    else
        logAndPrint "ERR_NO:${FILE_NOT_EXIST}; The ${_COMMON_INC_FILE} not exists."
        exit 1
    fi
    return
}

function show_relation () {
    relation_situation="$1"
    req_pkg_name="$2"
    req_pkg_path="$3"
    if [[ "$relation_situation" == "SUCC" ]] ;then
        logAndPrint "[INFO]relationship of opp with ${req_pkg_name} in path ${req_pkg_path} check success"
    else
        logAndPrint "[WARN]relationship of opp with ${req_pkg_name} in path ${req_pkg_path} check failed."
    fi
    return
}

function find_version_check(){
    if [[ "$(id -u)" != "0" ]]; then
        ccec_compiler_path=$(find ${HOME} -name "ccec_compiler")
    else
        ccec_compiler_path=$(find /usr/local -name "ccec_compiler")
    fi
    check_opp_version_file
    ret_check_opp_version_file=$opp_ver_info
    for var in ${ccec_compiler_path[@]}
        do
        run_pkg_path=$(dirname "${var}")
# find run pkg name
        select_last_dir_component "${run_pkg_path}"
        ret_pkg_name=$last_component
#get check version
        check_version_file "${run_pkg_path}" "${ret_pkg_name}"
        ret_check_version_file=$version_file
#check relation
        check_relation "${ret_check_opp_version_file}" "${ret_pkg_name}" "${ret_check_version_file}"
        ret_check_relation=$ret_situation
#show relation
        show_relation "${ret_check_relation}" "${ret_pkg_name}" "${run_pkg_path}"
         done
    return
}


function path_version_check(){
    path_env_list="$1"
    check_opp_version_file
    ret_check_opp_version_file=$opp_ver_info
    path_list=`echo "${path_env_list}" | cut -d"=" -f2 `
    array=(${path_list//:/ })
    for var in ${array[@]}
    do
        path_ccec_compile=$(echo ${var} | grep -w "ccec_compiler")
        if [[ "${path_ccec_compile}" != "" ]]; then
            pkg_path=$(dirname $(dirname "${path_ccec_compile}"))
# find run pkg name
            select_last_dir_component "${pkg_path}"
            ret_pkg_name=$last_component
#get check version
            check_version_file "${pkg_path}" "${ret_pkg_name}"
            ret_check_version_file=$version_file
#check relation
            check_relation "${ret_check_opp_version_file}" "${ret_pkg_name}" "${ret_check_version_file}"
            ret_check_relation=$ret_situation
#show relation
            show_relation "${ret_check_relation}" "${ret_pkg_name}" "${pkg_path}"
        else
            echo "the var_case does not contains ccec_compiler" >> /dev/null 2
        fi
    done
    return
}

#get the dir of xxx.run
opp_install_path_curr=`echo "$2" | cut -d"/" -f2- `

# cut first two params from *.run
i=0
while true
do
    if [[ x"$1" == x"" ]]; then
        break
    fi
    if [[ "`expr substr "$1" 1 2 `" == "--" ]]; then
       i=`expr $i + 1`
    fi
    if [[ $i -gt 2 ]]; then
        break
    fi
    shift 1
done

if [[ "$@" == "" ]]; then
    echo "ERR_NO:${PARAM_INVALID};ERR_DES:Unrecognized parameters. \
Try './xxx.run --help for more information.'"
    exit 1
fi



# init install cmd status, set default as n
in_cmd_list="$*"
is_uninstall=n
is_install=n
is_upgrade=n
is_quiet=n
is_viewlog=n
is_input_path=n
is_check=n
in_install_type=""
in_install_path=""
in_username=""
in_usergroup=""

iter_i=0
while true
do
    # skip 2 parameters avoid run pkg and directory as input parameter
    case "$1" in
    --version)
        if [[ -e "${_VERSION_INFO_FILE}" ]]; then
            . "${_VERSION_INFO_FILE}"
            echo ${Version}
            exit 0
        else
            echo "ERR_NO:${FILE_NOT_EXIST};ERR_DES:The version file \
(${_VERSION_INFO_FILE}) not exists or without execute permission."
            exit 1
        fi
        ;;
    --run | --full | --devel)
        in_install_type=$(echo ${1} | awk -F"--" '{print $2}')
        is_install=y
        iter_i=$(( ${iter_i} + 1 ))
        shift
        ;;
    --upgrade)
        is_upgrade=y
        shift
        ;;
    --uninstall)
        is_uninstall=y
        shift
        ;;
    --install-path=*)
        is_input_path=y
        in_install_path=`echo ${1} | cut -d"=" -f2 `
        # empty patch check
        if [[ "${in_install_path}" == "" ]]; then
            echo "[opp] ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path \
not support that the install path is empty."
            exit 1
        fi
        # space check
        if [[ "${in_install_path}" == *" "* ]]; then
            echo "[opp] ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path \
not support that the install path contains space character."
            exit 1
        fi
# check is ./or /
        if [[ "${in_install_path}" != "/"* ]] && [[ "${in_install_path}" == "./"* ]]; then
#relative location paths ./
            in_install_path=`echo ${in_install_path} | cut -d"." -f2- `
            in_install_path=$(matchFullpath "${in_install_path}")
        fi
#relative location paths ../
        if [[ "${in_install_path}" == "../"* ]]; then
            in_install_path=$(matchFullpath "${in_install_path}")
        fi
#absolute location paths
        if [[ "${in_install_path}" == "/"* ]]; then
            in_install_path="${in_install_path}"
        fi
        # path must start with the root dir
        if [[ "${in_install_path}" != "/"* ]]; then
            echo "[opp] ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path \
must with absolute path that which is start with root directory /. Such as --install-path=/${in_install_path}"
            exit 1
        fi
        # fliter the last "/" character
        in_install_path=`echo ${in_install_path} | sed "s/\/*$//g"`
        if [[ ${in_install_path} == "" ]]; then
            in_install_path="/"
        fi

        isOnlyLastNotExistDir "${in_install_path}"
        ret_isOnlyLastNotExistDir_path=$isOnlyLastNotExistDir_path
        ret_The_Num_Last_Not_ExistDir=$The_Num_Last_Not_ExistDir
        shift
        ;;
    --install-username=*)
        in_username=`echo $1 | cut -d"=" -f2 `
        shift
        ;;
    --install-usergroup=*)
        in_usergroup=`echo $1 | cut -d"=" -f2 `
        shift
        ;;
    --quiet)
        is_quiet=y
        shift
        ;;
    --install-for-all)
        is_for_all=y
        shift
        ;;
    --check)
        is_check=y
        shift
        ;;
    --check-path=*)
        check_path=$1
        shift
        ;;
    -*)
        echo "[opp] ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameters [$1], \
operation execute failed. Please use [--help] to see the useage."
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

if [[ "${iter_i}" == 2 ]]; then
    echo "[opp] ERR_NO:${PARAM_INVALID};ERR_DES:only support one type: full/run/devel, operation failed!"
    exit 1
fi

# must init target install path first before installation
if [[ "${is_input_path}" != y ]]; then
    _TARGET_INSTALL_PATH="${_DEFAULT_INSTALL_PATH}"
else
     _TARGET_INSTALL_PATH="${in_install_path}"
fi
_UNINSTALL_SHELL_FILE="${_TARGET_INSTALL_PATH}""/opp/script/opp_uninstall.sh"
# adpter for old version's path
if [[ ! -f "${_UNINSTALL_SHELL_FILE}" ]]; then
    _UNINSTALL_SHELL_FILE="${_TARGET_INSTALL_PATH}""/opp/scripts/opp_uninstall.sh"
fi

# init log file path before installation
_INSTALL_INFO_FILE="${_TARGET_INSTALL_PATH}${_INSTALL_INFO_SUFFIX}"
if [[ ! -f "${_INSTALL_INFO_FILE}" ]]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
fi

if [[ "$(id -u)" != "0" ]]; then
    _LOG_PATH_AND_FILE_GROUP="root"
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

# creat log folder and log file
if [[ ! -d "${_LOG_PATH}" ]]; then
    mkdir -p "${_LOG_PATH}"
fi
if [[ ! -f "${_INSTALL_LOG_FILE}" ]]; then
    touch "${_INSTALL_LOG_FILE}"
fi
if [[ ! -f "${_OPERATE_LOG_FILE}" ]]; then
    touch "${_OPERATE_LOG_FILE}"
fi


logAndPrint "[INFO]:Execute the opp run package."
logAndPrint "[INFO]:OperationLogFile path: ${_INSTALL_LOG_FILE}."
logAndPrint "[INFO]:Input params: $in_cmd_list"

if [[ "${is_check}" == "y" ]] && [[ "${check_path}" == "" ]]; then
    path_env_list=$(env | grep -w PATH)
    path_ccec_compile=$(echo ${path_env_list} | grep -w "ccec_compiler")
    if [[ "${path_ccec_compile}" != "" ]]; then
        path_version_check "${path_env_list}"
    else
        find_version_check
    fi
    exit 0
fi

if [[ "${is_check}"=="y" ]] && [[ "${check_path}" != "" ]]; then
    _VERCHECK_FILE="${_CURR_PATH}""/ver_check.sh"
    if [[ ! -f "${_VERCHECK_FILE}" ]]; then
        logAndPrint "ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_VERCHECK_FILE}) not exists. Please make sure that the opp module\
 installed in (${_VERCHECK_FILE}) and then set the correct install path."
    fi
    bash "${_VERCHECK_FILE}" "${check_path}"
    exit 0
fi

# os check
#checkOSAndArch
# if [[ "$?" != 0 ]]; then
#     logOperationRetStatus "$(getOperationName)" "$(getOperationInstallType)" "1" "${in_cmd_list}"
#     exit 1
# fi

# general user cant specific the install user and group
checkEmptyUserAndGroup "${in_username}" "${in_usergroup}"
if [[ "$?" != 0 ]]; then
    logOperationRetStatus "$(getOperationName)" "$(getOperationInstallType)" "1" "${in_cmd_list}"
    exit 1
fi

# installed version check and print
installed_version=$(getInstalledInfo "${KEY_INSTALLED_VERSION}")
runpkg_version=$(getRunpkgInfo "${KEY_RUNPKG_VERSION}")
installed_user=$(getInstalledInfo "${KEY_INSTALLED_UNAME}")
installed_group=$(getInstalledInfo "${KEY_INSTALLED_UGROUP}")
if [[ "${installed_version}" == "" ]]; then
    logAndPrint "[INFO]:Version of installing opp module is ${runpkg_version}."
else
    if [[ "${runpkg_version}" != "" ]]; then
        logAndPrint "[INFO]:Existed opp module version is ${installed_version}, \
the new opp module version is ${runpkg_version}."
    fi
fi

# get installed version's user and group
if [[ "${installed_version}" != "" ]] && [[ "${is_uninstall}" != "y" ]] && [[ "${is_upgrade}" != "y" ]]; then
    if [[ "${in_username}" != "" ]] && [[ "${in_usergroup}" != "" ]]; then
        if [[ "${installed_user}" != "${in_username}" ]] || [[ "${installed_group}" != "${in_usergroup}" ]]; then
            logAndPrint "[ERROR]ERR_NO:0x0095;ERR_DES:The user and group are not same with last installation,do not support overwriting installation!"
            exit 1
        fi
    else
        in_username=${installed_user}
        in_usergroup=${installed_group}
    fi
fi

# input username and usergroup valid check
if [[ "${in_username}" == "" ]] && [[ "${in_usergroup}" == "" ]]; then
    _TARGET_USERNAME="${_CURR_OPERATE_USER}"
    _TARGET_USERGROUP="${_CURR_OPERATE_GROUP}"
elif [[ "${in_username}" != "" ]] && [[ "${in_usergroup}" != "" ]]; then
    _TARGET_USERNAME="${in_username}"
    _TARGET_USERGROUP="${in_usergroup}"
    _DEFAULT_USERNAME="${in_username}"
    _DEFAULT_USERGROUP="${in_usergroup}"
else
    logAndPrint "ERR_NO:${PARAM_INVALID};ERR_DES:\
Only input user name or user group is invalid for install operations."
    logOperationRetStatus "$(getOperationName)" "$(getOperationInstallType)" "1" "${in_cmd_list}"
fi

#Support the installation script when the specified path (relative path and absolute path) does not exist
if [[ "${is_input_path}" == "y" ]];then
    checkGroupValidWithUser "${_TARGET_USERGROUP}" "${_TARGET_USERNAME}"
    if [[ "${ret_isOnlyLastNotExistDir_path}" == "" ]]; then
#Penultimate path not exists
        logAndPrint "ERR_NO:${FILE_NOT_EXIST}; The directory:${ret_The_Num_Last_Not_ExistDir} not exist, please creat this directory."
        exit 1
    else
        if [[ -d "${in_install_path}" ]]; then
            checkLastFoldersPermissionforcheckpath "${in_install_path}" "${_TARGET_USERNAME}"
            if [[ "$?" == 0 ]]; then
#All paths exist with write permission
                in_install_path=${in_install_path}
            else
#All paths exist, no write permission
                exit 1
            fi
        else
            checkLastFoldersPermissionforcheckpath "${ret_isOnlyLastNotExistDir_path}" "${_TARGET_USERNAME}"
            if [[ "$?" == 0 ]]; then
#penultimate path exists with write permission
                if [[ "${in_install_path}" == "${_DEFAULT_INSTALL_PATH}" ]] && [[ ! -d "${_DEFAULT_INSTALL_PATH}" ]] && [[ "$(id -u)" == "0" ]]; then
                    creat_checkpath "${in_install_path}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
                    in_install_path=${in_install_path}
                else
                    creat_checkpath "${in_install_path}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
                    in_install_path=${in_install_path}
                fi
            else
#Penultimate path exists, no write permission
                exit 1
            fi
        fi
    fi
fi

if [[ "${is_install}" == "y" ]];then
    # devel mode no need set the correct install user
    if [[ "${in_install_type}" != "devel" ]]; then
        checkUserGroupRelationValid "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
        if [[ "$?" != 0 ]]; then
            logAndPrint "ERR_NO:${INSTALL_FAILED};ERR_DES:Installation of opp\
 module execute failed!"
            logOperationRetStatus "$(getOperationName)" "$(getOperationInstallType)" "1" "${in_cmd_list}"
        fi
    else
        checkUserExist "${_TARGET_USERNAME}"
        devel_user_ret="$?"
        checkGroupValidWithUser "${_TARGET_USERGROUP}" "${_TARGET_USERNAME}"
        devel_group_ret="$?"
        if [[ "${devel_user_ret}" != "0" ]] || [[ "${devel_group_ret}" != "0" ]]; then
            _TARGET_USERNAME="${_DEFAULT_USERNAME}"
            _TARGET_USERGROUP="${_DEFAULT_USERGROUP}"
            if [[ "${is_quiet}" == "y" ]]; then
                logAndPrint "[WARNING]:Input user name or user group is invalid. \
Would you like to set as the default user name (${_DEFAULT_USERNAME}),\
user group (${_DEFAULT_USERGROUP}) for devel mode? y"
            else
                logAndPrint "[WARNING]:Input user name or user group is invalid. \
Would you like to set as the default user name (${_DEFAULT_USERNAME}),\
user group (${_DEFAULT_USERGROUP}) for devel mode? [y/n]"
                while true
                do
                    read yn
                    if [[ "$yn" == "n" ]]; then
                        logAndPrint "[INFO]:Exit to install opp module."
                        logOperationRetStatus "$(getOperationName)" "$(getOperationInstallType)" "1" "${in_cmd_list}"
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        echo "[WARNING]:Input error, please input y or n to choose!"
                    fi
                done
            fi
        fi
    fi
    # install need check whether the custom user can access the folders or not
    if [[ $(id -u) -ne 0 ]]; then
        checkPreFoldersPermission "${_TARGET_INSTALL_PATH}/opp" "${_TARGET_USERNAME}"
    fi
    if [[ "$?" != 0 ]]; then
        logOperationRetStatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi

    # use uninstall to clean the install folder
    precleanBeforeInstall "${_TARGET_INSTALL_PATH}"
    if [[ "$?" != 0 ]]; then
        logOperationRetStatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi

    _FIRST_NOT_EXIST_DIR=$(getFirstNotExistDir "${_TARGET_INSTALL_PATH}/opp")

    is_the_last_dir_opp=""
    if [[ "${_FIRST_NOT_EXIST_DIR}" == "${_TARGET_INSTALL_PATH}/opp" ]]; then
        is_the_last_dir_opp=1
    else
        is_the_last_dir_opp=0
    fi
    # call opp_install.sh
    bash "${_INSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_DEFAULT_USERNAME}" "${_DEFAULT_USERGROUP}" "${in_install_type}" "${is_quiet}" "${_FIRST_NOT_EXIST_DIR}" "${is_the_last_dir_opp}" "${is_for_all}"
    if [[ "$?" != 0 ]]; then
        logOperationRetStatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi
    chmod -R 500 "${_TARGET_INSTALL_PATH}/opp/script" 2> /dev/null
    if [ $(id -u) -eq 0 ]; then
        chown -R "root":"root" "${_TARGET_INSTALL_PATH}/opp/script" 2> /dev/null
        chown "root":"root" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
    else
        chmod 750 "${_TARGET_INSTALL_PATH}" 2> /dev/null
    fi

    logOperationRetStatus "install" "${in_install_type}" "$?" "${in_cmd_list}"
fi

if [[ "${is_upgrade}" == "y" ]];then
    install_type=$(getInstalledInfo "${KEY_INSTALLED_TYPE}")
    # upgrade not support specific username and usergroup
    checkEmtpyUser "upgrade" "${in_username}" "${in_usergroup}"
    if [[ "$?" != 0 ]]; then
        logOperationRetStatus "upgrade" "${install_type}" "1" "${in_cmd_list}"
    fi
    # call opp_upgrade.sh
    bash "${_UPGRADE_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_DEFAULT_USERNAME}" "${_DEFAULT_USERGROUP}" "${is_quiet}"
    chmod -R 500 "${_TARGET_INSTALL_PATH}/opp/script" 2> /dev/null
    if [ $(id -u) -eq 0 ]; then
        chown -R "root":"root" "${_TARGET_INSTALL_PATH}/opp/script" 2> /dev/null
        chown "root":"root" "${_TARGET_INSTALL_PATH}/opp" 2> /dev/null
    else
        chmod 750 "${_TARGET_INSTALL_PATH}" 2> /dev/null
    fi

    logOperationRetStatus "upgrade" "${install_type}" "$?" "${in_cmd_list}"
fi

if [[ ${is_uninstall} == "y" ]];then
    install_type=$(getInstalledInfo "${KEY_INSTALLED_TYPE}")
    # uninstall not support specific username and usergroup
    checkEmtpyUser "uninstall" "${in_username}" "${in_usergroup}"
    if [[ "$?" != 0 ]]; then
        logOperationRetStatus "uninstall" "${install_type}" "1" "${in_cmd_list}"
    fi
    if [[ ! -f "${_UNINSTALL_SHELL_FILE}" ]]; then
        logAndPrint "ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_UNINSTALL_SHELL_FILE}) not exists. Please make sure that the opp module\
 installed in (${_TARGET_INSTALL_PATH}) and then set the correct install path."
        logOperationRetStatus "uninstall" "${install_type}" "1" "${in_cmd_list}"
    fi
    # call opp_uninstall.sh
    bash "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "uninstall" "${is_quiet}"
    logOperationRetStatus "uninstall" "${install_type}" "$?" "${in_cmd_list}"
fi
exit 0
