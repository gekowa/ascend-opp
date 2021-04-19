#!/bin/bash

file_list=""
cp_list=""
mkdir_list=""
del_list=""

# 修改各文件及目录的权限
change_mod() {
    local install_for_all="$3"
    local mod=$2
    # 对于软连接，可能目标文件还没有拷贝进来，导致无法修改mod，这里过滤掉软连接
    if [ -L "$1" ]; then
        return 0
    fi
    if [ "$2" != "NA" ]; then
        # 如果设置了install_for_all，则安装时other权限跟group权限对齐
        if [ "$install_for_all" = "y" ]; then
            local new_mod=${mod:0:-1}${mod: -2:1}
            chmod "$new_mod" "$1"
            return 0
        fi
        chmod "$2" "$1"
        if [ $? -ne 0 ]; then
            echo "Error: $1 chmod failed!"
            exit 1
        fi
    fi
}

# 修改各文件及目录的属性
change_own() {
    if [ "$2" != "NA" ]; then
        if [ "$2" = "\$username:\$usergroup" ]; then
            chown "$username:$usergroup" "$1"
        else
            chown "$2" "$1"
        fi
        if [ $? -ne 0 ]; then
            echo "Error: $1 chown failed!"
            exit 1
        fi
    fi
}

# 修改权限和属组
change_mod_and_own(){
    target="$1"
    mod="$2"
    own="$3"
    local install_for_all="$4"
    change_mod "$target" "$mod" "$install_for_all"
    change_own "$target" "$own"
}

change_dir_mod() {
    if [ "$2" != "NA" ]; then
        chmod "$2" "$1"
        if [ $? -ne 0 ]; then
            echo "Error: $1 chmod failed!"
            exit 1
        fi
    fi
}

# 修改各文件及目录的属性
change_dir_own() {
    if [ "$2" != "NA" ]; then
        if [ "$2" = "\$username:\$usergroup" ]; then
            chown "$username:$usergroup" "$1"
        else
            chown "$2" "$1"
        fi
        if [ $? -ne 0 ]; then
            echo "Error: $1 chown failed!"
            exit 1
        fi
    fi
}

#创建软连接
create_softlink() {
    if [ "$2" != "NA" ]; then
        if [ ! -d "$(dirname "$2")" ]; then
            mkdir -p "$(dirname "$2")"
        fi
        link_target=$(
            cd "$(dirname "$1")" || return
            pwd
        )/$(basename "$1")
        ln -sf "$link_target" "$2"
        if [ $? -ne 0 ]; then
            echo "Error: $2 softlink to $link_target failed!"
            exit 1
        fi
    fi
}

# 创建文件夹
create_folder() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        if [ $? -ne 0 ]; then
            echo "Error: $1 mkdir failed!"
            exit 1
        fi
    fi
    if [ $# -ge 4 ]; then
        create_softlink "$1" "$4"
    fi
}

# 拷贝文件
copy_file() {
    if [ -e "$1" ]; then
        cp -rf "$1" "$(dirname "$2")"
        if [ $? -ne 0 ]; then
            echo "Error: $1 copy failed!"
            exit 1
        fi
        if [ $# -ge 5 ]; then
            create_softlink "$2" "$5"
        fi
    else
        echo "Error: $1 is not existed!"
        exit 1
    fi
}

#移除文件
remove_file() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        rm -f "$1"
        if [ $? -ne 0 ]; then
            echo "Error: $1 remove failed!"
            exit 1
        fi
    fi
    if [ -L "$2" ]; then
        rm -f "$2"
        if [ $? -ne 0 ]; then
            echo "Error: $2 remove failed!"
            exit 1
        fi
    fi
}

#移除文件夹
remove_dir() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        rm -fr "$1"
        if [ $? -ne 0 ]; then
            echo "Error: $1 remove failed!"
            exit 1
        fi
    fi
}

#恢复目录的权限
resetmod() {
    if [ -d "$1" ]; then
        a=777
        b=$(umask)
        val=$(expr $a - $b)
        if [ $2 != "NA" ]; then
            change_dir_mod "$1" $val
        fi
    fi
}

#解析filelist.csv脚本,入参为$1: install_type $2:operate_type $3:filelist_path
# 输出 $3,$4,$6,$7,$9,$12,$13 即：relative_path_in_pkg,relative_install_path,permission,owner:group,softlink,configurable,hash
parse_filelist() {
    local install_type="$1"
    local operate_type="$2"
    local filelist_path="$3"
    local feature_type="$4"

    if [ "$install_type" != "full" ]; then
        install_type_list="all $install_type"
    else
        install_type_list="all docker devel run"
    fi
    if [ "$feature_type" != "all" ]; then
        feature_list=("$(echo $feature_type | tr ',' ' ')")
        feature_type_list="all ${feature_list[*]}"
    else
        feature_type_list="all"
    fi
    operate_type_list="$operate_type"

    if [ ! -f "$filelist_path" ]; then
        echo "Error: $filelist_path is not existed!"
        exit 1
    fi
    filelist=$(awk -v install_type_list="$install_type_list" -v operate_type_list="$operate_type_list" -v feature_type_list="$feature_type_list" '
        BEGIN{
            FS= ",";
        }
        {
            #feature 只需要有一个匹配到就行了
            matched_feature_type=1
            if($10 != "all" && feature_type_list != "all" )
            {
                matched_feature_type=0
                split($10,features,";")
                for(f in features)
                {
                    if(feature_type_list ~ features[f])
                    {
                        matched_feature_type=1
                        break;
                    }
                }
            }

            #dependency必须都要匹配到
            matched_dependency=1;
            if($11 != "NA" && feature_type_list != "all" )
            {
                split($11,dependency,";")
                for(f in dependency)
                {
                    if(feature_type_list ~ dependency[f])
                    {
                        continue;
                    }
                    else
                    {
                        matched_dependency=0;
                        break;
                    }
                }
            }

            matched_install_type=0
            split($8,install_types,";");
            for (i in install_types){
                if(install_type_list  ~  install_types[i]){
                    matched_install_type=1;
                    break;
                }
            }
            if( (operate_type_list ~ $2) && ( matched_feature_type==1 ) && (matched_dependency==1)  && (matched_install_type==1))
                print $3,$4,$6,$7,$9,$12,$13
        }' "$filelist_path")

}

ProgressBar() {
    local current=$1
    local total=$2
    local now=$((current * 100 / total))
    local last=$(((current - 1) * 100 / total))
    [[ $((last % 2)) -eq 1 ]] && let last++
    local str=$(for i in $(seq 1 $((last / 2))); do printf '#'; done)
    for ((i = $last; $i <= $now; i += 2)); do
        printf "\r[%-50s]%d%%" "$str" $i
        str+='#'
    done
}

#执行拷贝动作，入参为$1: install_type, $2: install_path $3:filelist_path
do_copy_files() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local feature_type="$4"
    parse_filelist "$install_type" "copy " "$filelist_path" "$feature_type"

    line_count=$(echo "$filelist" | wc -l)

    echo "copying install files..."
    n=0
    echo "$filelist" | while read line; do
        array=($line)
        target=${array[1]}
        softlink=${array[4]}
        configurable=${array[5]}
        hash_value=${array[6]}

        if [[ "$target" != /* ]]; then
            target="$install_path/$target"
        fi
        tmpdir=$(dirname "$target")
        if [ ! -d "$tmpdir" ]; then
            mkdir -p "$tmpdir"
        fi
        if [ "$softlink" != "NA" ] && [[ "$softlink" != /* ]]; then
            softlink="$install_path/$softlink"
        fi
        
        # 如果目标文件已经存在，而且是配置文件，则不执行覆盖操作
        if [[ -e "$target" ]] && [[ $configurable = 'TRUE' ]]; then
            continue
        fi

        copy_file "${array[0]}" "$target" "${array[2]}" "${array[3]}" "$softlink"
        # n=$(($n + 1))
        # if [[ $(($n % 10)) = 0 || $n = $line_count ]]; then
        #     #ProgressBar $n $line_count
        # fi
    done
    echo "copy install files successfully!"
}

#执行创建目录动作，入参为$1: install_type, $2: install_path $3:filelist_path
do_create_dirs() {
    local install_type="$2"
    local install_path="$3"
    local filelist_path="$4"
    local feature_type="$5"
    parse_filelist "$install_type" "mkdir" "$filelist_path" "$feature_type"

    #echo "creating install folders..."

    echo "$filelist" | while read line; do
        array=($line)
        target=${array[1]}
        if [[ "$target" != /* ]]; then
            target="$install_path/$target"
        fi
        mod="NA"
        own="NA"
        soflink="NA"
        if [ "$1" = "all" ] || [ "$1" = "chmod" ] || [ "$1" = "resetmod" ]; then
            mod=${array[2]}
            own=${array[3]}
        fi
        if [ "$1" = "all" ] || [ "$1" = "mkdir" ]; then
            soflink=${array[4]}
        fi

        if [ "$1" = "resetmod" ]; then
            resetmod "$target" "$mod"
        else
            if [ -L "$target" ] ; then
                rm -f "$target"
                echo "$target is an existing soft-link, deleted."
            fi
            create_folder "$target" "$mod" "$own" "$soflink"
        fi
    done
    echo "create install folders successfully!"
}

#删除安装文件，入参为$1: install_type, $2: install_path $3:filelist_path
remove_install_files() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local feature_type="$4"
    parse_filelist "$install_type" "copy del" "$filelist_path" "$feature_type"

    echo "deleting install files..."
    echo "$filelist" | while read line; do
        array=($line)
        target=${array[1]}
        softlink=${array[4]}
        configurable=${array[5]}
        hash_value=${array[6]}

        if [ "$target" != "NA" ]; then
            if [[ "$target" != /* ]]; then
                target="$install_path/$target"
            fi
            if [ "$softlink" != "NA" ] && [[ "$softlink" != /* ]]; then
                softlink="$install_path/$softlink"
            fi
            if [ -d "$target" ]; then
                #echo "$target is directory,skip!"
                continue
            fi
            mod=${array[2]}
            if [ $configurable = 'TRUE' ]; then
                echo "$hash_value $target" | sha256sum --check &>/dev/null
                if [ $? -ne 0 ]; then
                    echo "$target has been modified!"
                    continue
                fi
            fi
            remove_file "$target" "$softlink" "$mod" &
        fi
    done
    wait
    echo "remove install files successfully!"
}

#删除安装文件夹，入参为$1: install_type, $2: install_path $3:filelist_path
remove_install_dirs() {
    local install_type="$1"
    local install_path="$2"
    local filelist_path="$3"
    local feature_type="$4"
    parse_filelist "$install_type" "mkdir" "$filelist_path" "$feature_type"

    #对第二列文件路径做倒序排列，保证先删除子文件夹，再删除父文件夹
    sort_filelist=$(echo "$filelist" |sort -k2 -r)

    echo "deleting installed folders..."
    echo "$sort_filelist" | while read line; do
        array=($line)
        target=${array[1]}
        if [ "$target" != "NA" ]; then
            if [[ "$target" != /* ]]; then
                target="$install_path/$target"
            fi
            if [ ! -d "$target" ]; then
                #echo "dir $target not exists!"
                continue
            fi
            if [ "$(ls -A "$target")" ]; then
                #echo "$target not empty,will not remove!"
                continue
            fi
            #echo "delete $target"
            remove_dir "$target"
        fi
    done
    echo "remove install folders successfully!"
}

#修改文件的权限和属组，入参为$1: install_type, $2: install_path $3:filelist_path
do_change_mod_and_own_files() {
    local install_type="$2"
    local install_path="$3"
    local filelist_path="$4"
    local feature_type="$5"
    local install_for_all="$6"
    parse_filelist "$install_type" "copy del" "$filelist_path" "$feature_type"

    echo "change mod install files ..."
    echo "$filelist" | while read line; do
        array=($line)
        target=${array[1]}
        if [[ "$target" != /* ]]; then
            target="$install_path/$target"
        fi
        if [ -d "$target" ]; then
            #echo "$target is directory,skip!"
            continue
        fi
        mod=${array[2]}
        own=${array[3]}
        change_mod_and_own "$target" "$mod" "$own" "$install_for_all" &
    done
    wait
    echo "change mod install files successfully!"
}

#修改目录的权限和属组，入参为$1: install_type, $2: install_path $3:filelist_path
do_change_mod_and_own_dirs() {
    local install_type="$2"
    local install_path="$3"
    local filelist_path="$4"
    local feature_type="$5"
    local install_for_all="$6"
    parse_filelist "$install_type" "mkdir" "$filelist_path" "$feature_type"

    #对第二列文件路径做倒序排列，保证先修改子文件夹，再修改父文件夹
    sort_filelist=$(echo "$filelist" |sort -k2 -r)
    echo "change mod install dirs ..."
    echo "$sort_filelist" | while read line; do
        array=($line)
        target=${array[1]}
        if [[ "$target" != /* ]]; then
            target="$install_path/$target"
        fi
        if [ ! -d "$target" ]; then
            #echo "$target is directory,skip!"
            continue
        fi
        mod=${array[2]}
        own=${array[3]}
        change_mod_and_own "$target" "$mod" "$own" "$install_for_all"
    done
    echo "change mod install dirs successfully!"
}

#打印安装信息
print_install_content() {
    local install_type="$1"
    local filelist_path="$2"
    local feature_type="$3"

    parse_filelist "$install_type" "copy mkdir" "$filelist_path" "$feature_type"

    echo "$filelist" | while read line; do
        echo $line
    done
    exit 0
}

help_info() {
    echo "Usage: $0 [param][ --username=<user> --usergroup=<group> ] install_tpye install_path filelist_path"
    echo "param can be one of the following :"
    echo "    --help       | -h      : Print out this help message"
    echo "    --copy       | -c      : Copy the install content"
    echo "    --mkdir      | -m      : Create the install folder, and set dir right"
    echo "    --makedir    | -d      : Create the install folder, not set dir right"
    echo "    --chmoddir   | -o      : Set dir right"
    echo "    --restoremod | -e      : Restore dir right"
    echo "    --remove     | -r      : Remove the install content"
    echo "    --print      | -p      : print the install content and folder info"
    echo "    --username=<user>    : specify user"
    echo "    --usergroup=<group>  : specify group"
    echo "    --install_for_all      : Install for all user"
    exit 0
}

OPERATE_TYPE=""

while true; do
    case "$1" in
    --copy | -c)
        OPERATE_TYPE="copy"
        shift
        ;;
    --mkdir | -m)
        OPERATE_TYPE="mkdir"
        shift
        ;;
    --makedir | -d)
        OPERATE_TYPE="makedir"
        shift
        ;;
    --chmoddir | -o)
        OPERATE_TYPE="chmoddir"
        shift
        ;;
    --restoremod | -e)
        OPERATE_TYPE="restoremod"
        shift
        ;;
    --remove | -r)
        OPERATE_TYPE="remove"
        shift
        ;;
    --print | -p)
        OPERATE_TYPE="print"
        shift
        ;;
    --username=*)
        username=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --usergroup=*)
        usergroup=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --install_for_all)
        install_for_all="y"
        shift
        ;;
    -h | --help)
        help_info
        ;;
    -*)
        echo Unrecognized input options : "$1"
        help_info
        ;;
    *)
        break
        ;;
    esac
done

if [ $# -lt 3 ]; then
    echo "It's too few input params: $*"
    exit 1
else
    if [ $# == 3 ]; then
        input_feature="all"
    else
        input_feature="$4"
    fi
    case $OPERATE_TYPE in
    "copy")
        do_copy_files "$1" "$2" "$3" "$input_feature"
        ;;
    "remove")
        filelist_path="$3"
        tmp_root=/tmp
        if ! test -d "$tmp_root"; then
            tmp_root="$PWD"
        fi
        if ! test -d "$tmp_root"; then
            tmp_root="$HOME"
        fi
        tmp_filelist_path=$(mktemp "$tmp_root/filelist_XXXX.csv" || exit 1)
        cp -f "$filelist_path" "$tmp_filelist_path"
        remove_install_files "$1" "$2" "$tmp_filelist_path" "$input_feature"
        remove_install_dirs "$1" "$2" "$tmp_filelist_path" "$input_feature"
        rm -f "$tmp_filelist_path"
        ;;
    "mkdir")
        do_create_dirs "all" "$1" "$2" "$3" "$input_feature"
        ;;
    "makedir")
        do_create_dirs "mkdir" "$1" "$2" "$3" "$input_feature"
        ;;
    "chmoddir")
        #do_create_dirs "chmod" "$1" "$2" "$3" "$input_feature"
        do_change_mod_and_own_files " " "$1" "$2" "$3" "$input_feature" "$install_for_all"
        do_change_mod_and_own_dirs " " "$1" "$2" "$3" "$input_feature" "$install_for_all"
        ;;
    "restoremod")
        do_create_dirs "resetmod" "$1" "$2" "$3" "$input_feature"
        ;;
    "print")
        print_install_content "$1" "$3" "$input_feature"
        ;;
    esac
fi

exit 0


