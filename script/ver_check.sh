#!/bin/bash

req_ver_path=$1
_CURR_PATH=$(dirname $(readlink -f $0))
_DEFAULT_INSTALL_PATH="/usr/local/Ascend"
FILE_NOT_EXIST="0x0080"

function getDate() {
    local _cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "${_cur_date}"
}

function logAndPrint() {
    echo "[opp] [$(getDate)] ""$1"
}

function check_path_pre (){
    in_checkpath_0="$1"
    in_checkpath_1=`echo ${in_checkpath_0} | cut -d"=" -f2 `
    if [[ "${in_checkpath_1}" == "" ]] ;then
        logAndPrint "[WARNING] please input correct path"
        exit 1
    fi
    arr=(`echo ${in_checkpath_1} | awk '{len=split($0,arr,",");for(i=1;i<=len;i++) print arr[i]}'`)
    len="${#arr[*]}"
    b=1
    for((i=0;i<"${len}";i++)); do
        select_last_dir_component "${arr[i]}"
        ret=$last_component
        if [[ "${ret}" != "" ]] ;then
            checked_path_temp[$b]=${arr[i]}
            ((b++))
        fi
    done
    checked_path="${checked_path_temp[*]}"
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
        ver_info="${_CURR_PATH}/../../version.info"
    elif [ -f "${_DEFAULT_INSTALL_PATH}/opp/version.info" ];then
        ver_info="${_DEFAULT_INSTALL_PATH}/opp/version.info"
    else
        logAndPrint "ERR_NO:${FILE_NOT_EXIST}; The [opp] version.info not exists."
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
    fi
    return
}

function show_relation () {
    relation_situation="$1"
    req_pkg_name="$2"
    req_pkg_path="$3"
    if [[ "$relation_situation" == "SUCC" ]] ;then
        logAndPrint "[INFO]relationship of opp with ${req_pkg_name} in path ${req_pkg_path} check success"
        return 0
    else
        logAndPrint "[WARNING]relationship of opp with ${req_pkg_name} in path ${req_pkg_path} check failed. \
do you want to continue.  [y/n] "
        while true
        do
        read yn
        if [[ "$yn" == "n" ]]; then
            echo "stop check!"
            exit 1;
        elif [ "$yn" = y ]; then
            break;
        else
            echo "WARN:Input error, please input y or n to choose!"
        fi
        done
  fi
}

function version_check () {
    path="$1"
#get opp version
    check_opp_version_file
    ret_check_opp_version_file=$ver_info
#get checked path
    check_path_pre "${path}"
    ret_check_path_pre=$checked_path
    if [[ "${ret_check_path_pre}" != "" ]] ;then
        for var in ${ret_check_path_pre[@]}
            do
            # select_last_dir_component "${var}"
            # component_ret=$last_component
#get atc or fwkacllib name
            select_last_dir_component "${var}"
            ret_last_component=$last_component
#get the version of atc/fwkacllib
            check_version_file "${var}" "${ret_last_component}"
            ret_check_version_file=$version_file
#check relation
            check_relation "${ret_check_opp_version_file}" "${ret_last_component}" "${ret_check_version_file}"
            ret_check_relation=$ret_situation
#show relation
            show_relation "${ret_check_relation}" "${ret_last_component}" "${var}"
            done
    fi
}

version_check "${req_ver_path}"
exit 0
